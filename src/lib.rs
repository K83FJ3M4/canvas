mod tile;
mod sort;
mod index;
mod alloc;

use std::num::NonZero;

use wgpu::{BindingResource, CommandEncoder, ComputePassDescriptor, Device, TextureFormat, TextureView};

use crate::alloc::Allocator;
use crate::index::{IndexBuffers, IndexPipeline};
use crate::sort::{SortBuffers, SortPipeline};
use crate::tile::{TileBuffers, TilePipeline};

pub struct DrawContext {
    index_pipeline: IndexPipeline,
    tile_pipeline: TilePipeline,
    sort_pipeline: SortPipeline,
    allocator: Allocator
}

pub struct DrawList {

}


impl DrawContext {
    pub fn new(device: Device, format: TextureFormat) -> DrawContext {
        DrawContext {
            allocator: Allocator::new(device.clone()),
            tile_pipeline: TilePipeline::new(device.clone(), format),
            sort_pipeline: SortPipeline::new(device.clone()),
            index_pipeline: IndexPipeline::new(device.clone())
        }
    }

    pub fn render(&mut self, encoder: &mut CommandEncoder, texture: TextureView, device: &Device, queue: &wgpu::Queue) {

        let list_ranges = self.allocator.alloc::<[u32; 2]>(32);
        let keys = self.allocator.alloc::<u32>(1024);
        let temp_keys = self.allocator.alloc::<u32>(keys.len());
        let lists = self.allocator.alloc::<u32>(keys.len());
        let temp_lists = self.allocator.alloc::<u32>(keys.len());
        let histogram_capacity = SortPipeline::min_histogram_capacity(keys.len());
        let histograms = self.allocator.alloc::<[u32; 16]>(histogram_capacity);
        let mut memory = self.allocator.finalize();

        let mut data = vec![0; 1024];
        data[0] = 1;
        data[1] = 3;
        data[15] = 30;
        data[31] = 1;
        memory.upload(encoder, keys, &data);
        let values_data = (0..1024).collect::<Vec<u32>>();
        memory.upload(encoder, lists, &values_data);

        if let Some(BindingResource::Buffer(list_ranges)) = memory.binding(list_ranges) {
            let size = list_ranges.size.map(NonZero::<u64>::get);
            encoder.clear_buffer(&list_ranges.buffer, list_ranges.offset, size);
        }

        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Canvas Compute Pass"),
            timestamp_writes: None
        });

        self.sort_pipeline.encode(&mut compute_pass, &mut memory, SortBuffers {
            keys,
            temp_keys,
            values: lists,
            temp_values: temp_lists,
            histograms,
        });

        self.index_pipeline.encode(&mut compute_pass, &mut memory, IndexBuffers {
            sorted_list_keys: keys,
            list_ranges: list_ranges,
        });

        if let Some(binding) = memory.binding(list_ranges) {
            if let wgpu::BindingResource::Buffer(binding) = binding {
                let slice = if let Some(size) = binding.size {
                    binding.buffer.slice(binding.offset..binding.offset + size.get())
                } else {
                    binding.buffer.slice(binding.offset..)
                };

                wgpu::util::DownloadBuffer::read_buffer(device, queue, &slice, |data| {
                    let Ok(buffer) = data else { return };
                    println!("\n");
                    println!("{:?}", bytemuck::cast_slice::<u8, [u32; 2]>(&buffer));
                });
            }
        }

        self.tile_pipeline.encode(&mut compute_pass, &mut memory, TileBuffers {
            list_ranges,
            texture,
            lists
        });
    }
}
