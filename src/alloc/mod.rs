use std::marker::PhantomData;
use std::mem::replace;
use std::num::NonZeroU64;

use bytemuck::Pod;
use wgpu::{BindingResource, Buffer, BufferAddress, BufferBinding, BufferDescriptor, BufferUsages, Device};

pub(super) struct Allocator {
    offset: usize,
    alignment: usize,
    buffer: Option<Buffer>,
    device: Device
}

#[derive(Clone, Copy)]
pub(super) struct AllocationMemory<'a> {
    buffer: &'a Buffer
}

#[derive(Clone, Copy)]
pub(super) struct Allocation<T: Pod> {
    marker: PhantomData<T>,
    offset: usize,
    len: usize
}

impl Allocator {
    pub(super) fn new(device: Device) -> Allocator {
        Allocator {
            offset: 0,
            alignment: device.limits()
                .min_storage_buffer_offset_alignment.max(1) as usize,
            buffer: None,
            device
        }
    }

    pub(super) fn alloc<T: Pod>(&mut self, len: usize) -> Allocation<T> {
        self.offset = self.offset.next_multiple_of(self.alignment);
        let offset = self.offset;
        let size = size_of::<T>() * len.max(1);
        self.offset += size;

        Allocation {
            marker: PhantomData,
            offset,
            len
        }
    }

    pub(super) fn finalize<'a>(&'a mut self, device: &Device) -> AllocationMemory<'a> {
        let size = replace(&mut self.offset, 0) as u64;
        self.buffer = self.buffer.take().filter(|buffer| buffer.size() >= size);
        let buffer = self.buffer.get_or_insert_with(|| device.create_buffer(&BufferDescriptor {
            label: None,
            size: size.checked_next_power_of_two().unwrap_or(u64::MAX).max(8),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false
        }));

        AllocationMemory { buffer }
    }
}

impl<T: Pod> Allocation<T> {
    pub(super) fn binding<'a>(&self, buffer: AllocationMemory<'a>) -> Option<BindingResource<'a>> {
        if self.len == 0 { return None }
        Some(BindingResource::Buffer(BufferBinding {
            buffer: buffer.buffer,
            offset: self.offset as BufferAddress,
            size: NonZeroU64::new((self.len * size_of::<T>()) as u64)
        }))
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }
}