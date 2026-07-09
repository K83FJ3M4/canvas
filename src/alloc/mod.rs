use std::iter::successors;
use std::marker::PhantomData;
use std::mem::replace;
use std::num::NonZeroU64;
use std::sync::mpsc::{Receiver, Sender, channel};

use bytemuck::{Pod, cast_slice};
use wgpu::{BindingResource, Buffer, BufferAddress, BufferBinding, BufferDescriptor, BufferUsages, CommandEncoder, Device, MapMode};

pub(super) struct Allocator {
    offset: usize,
    alignment: usize,
    buffer: Option<Buffer>,
    pools: Vec<StagingPool>,
    device: Device
}

pub(super) struct AllocationMemory<'a> {
    pools: &'a mut [StagingPool],
    buffer: &'a Buffer
}

#[derive(Clone, Copy)]
pub(super) struct Allocation<T: Pod> {
    marker: PhantomData<T>,
    offset: usize,
    len: usize
}

pub(super) struct StagingPool {
    sender: Sender<Buffer>,
    receiver: Receiver<Buffer>,
    free: Vec<(usize, Buffer)>,
    frame_index: usize,
    block_size: usize,
    device: Device
}

impl StagingPool {
    pub(super) fn new(device: Device, block_size: usize) -> StagingPool {
        let (sender, receiver) = channel();
        StagingPool {
            sender,
            receiver,
            block_size,
            free: Vec::new(),
            frame_index: 0,
            device
        }
    }

    pub(super) fn upload(&mut self, encoder: &mut CommandEncoder, data: &[u8], target: &Buffer, offset: u64) {
        assert!(data.len() <= self.block_size);
        let buffer = self.get_buffer();
        let range = 0..data.len() as u64;
        let mut view = buffer.get_mapped_range_mut(range).unwrap();
        view.copy_from_slice(data);
        drop(view);
        buffer.unmap();

        let length = data.len() as u64;
        let sender = self.sender.clone();
        encoder.copy_buffer_to_buffer(&buffer, 0, target, offset, length);
        encoder.map_buffer_on_submit(&buffer.clone(), MapMode::Write, .., move |result| {
            if result.is_ok() { sender.send(buffer).ok(); }
        });
    }

    fn get_buffer(&mut self) -> Buffer {
        self.receiver.try_recv().ok()
        .or_else(|| self.free.pop().map(|(.., buffer)| buffer))
        .unwrap_or_else(|| self.device.create_buffer(&BufferDescriptor {
            label: Some("Canvas Staging Buffer"),
            size: self.block_size as u64,
            usage: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
            mapped_at_creation: true
        }))
    }

    fn begin_frame(&mut self) {
        self.frame_index += 1;
        while let Ok(buffer) = self.receiver.try_recv() {
            self.free.push((self.frame_index, buffer));
        }
    }

    fn end_frame(&mut self) {
        self.free.retain(|(time, ..)| {
            self.frame_index.wrapping_sub(*time) < 32
        });
    }
}

impl Allocator {
    pub(super) fn new(device: Device) -> Allocator {
        let sizes = successors(Some(256 * 1024usize), |n| Some(n * 2));
        let mut pools = Vec::new();
        for block_size in sizes.take(6) {
            pools.push(StagingPool::new(device.clone(), block_size));
        }

        Allocator {
            offset: 0,
            alignment: device.limits()
                .min_storage_buffer_offset_alignment.max(4) as usize,
            buffer: None,
            device,
            pools
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

    pub(super) fn finalize<'a>(&'a mut self) -> AllocationMemory<'a> {
        let size = replace(&mut self.offset, 0) as u64;
        self.buffer = self.buffer.take().filter(|buffer| buffer.size() >= size);
        let buffer = self.buffer.get_or_insert_with(|| self.device.create_buffer(&BufferDescriptor {
            label: None,
            size: size.checked_next_power_of_two().unwrap_or(u64::MAX).max(8),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false
        }));

        for pool in self.pools.iter_mut() { pool.begin_frame() }
        AllocationMemory {
            pools: &mut self.pools,
            buffer
        }
    }
}

impl<'a> AllocationMemory<'a> {
    pub(super) fn binding<'b, T: Pod>(&'b self, allocation: Allocation<T>) -> Option<BindingResource<'b>> {
        if allocation.len == 0 { return None }
        Some(BindingResource::Buffer(BufferBinding {
            buffer: self.buffer,
            offset: allocation.offset as BufferAddress,
            size: NonZeroU64::new((allocation.len * size_of::<T>()) as u64)
        }))
    }

    pub(super) fn upload<T: Pod>(&mut self, encoder: &mut CommandEncoder, allocation: Allocation<T>, data: &[T]) {
        assert!(size_of::<T>() % 4 == 0);
        let bytes = cast_slice::<T, u8>(data);
        assert!(data.len() == allocation.len);
        let block_size = self.pools.last_mut().unwrap().block_size;
        let mut offset = allocation.offset as u64;

        for chunk in bytes.chunks(block_size) {
            for pool in self.pools.iter_mut() {
                if chunk.len() > pool.block_size { continue }
                pool.upload(encoder, chunk, self.buffer, offset);
                offset += chunk.len() as u64;
                break;
            }
        }
    }
}

impl<'a> Drop for AllocationMemory<'a> {
    fn drop(&mut self) {
        for pool in self.pools.iter_mut() {
            pool.end_frame();
        }
    }
}

impl<T: Pod> Allocation<T> {
    pub(super) fn len(&self) -> usize {
        self.len
    }
}