mod tile;
mod sort;
mod alloc;

use wgpu::{CommandEncoder, ComputePassDescriptor, Device, TextureFormat, TextureView};

use crate::alloc::Allocator;
use crate::sort::{SortBuffers, SortPipeline};
use crate::tile::TilePipeline;

pub struct DrawContext {
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
        }
    }

    pub fn render(&mut self, encoder: &mut CommandEncoder, texture: TextureView, device: &Device, queue: &wgpu::Queue) {

        let keys = self.allocator.alloc::<u32>(32);
        let temp_keys = self.allocator.alloc::<u32>(keys.len());
        let histograms = self.allocator.alloc::<[u32; 16]>(4);
        let mut memory = self.allocator.finalize();

        let mut data = vec![1000; 32];
        data[0] = 5;
        data[1] = 3;
        data[15] = 999999;
        data[31] = 49382;
        memory.upload(encoder, keys, &data);

        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Canvas Compute Pass"),
            timestamp_writes: None
        });

        self.sort_pipeline.encode(&mut compute_pass, &mut memory, SortBuffers {
            keys,
            temp_keys,
            histograms,
        });

        if let Some(binding) = memory.binding(keys) {
            if let wgpu::BindingResource::Buffer(binding) = binding {
                let slice = if let Some(size) = binding.size {
                    binding.buffer.slice(binding.offset..binding.offset + size.get())
                } else {
                    binding.buffer.slice(binding.offset..)
                };

                wgpu::util::DownloadBuffer::read_buffer(device, queue, &slice, |data| {
                    let Ok(buffer) = data else { return };
                    println!("\n");
                    println!("{:?}", bytemuck::cast_slice::<u8, u32>(&buffer));
                });
            }
        }

        self.tile_pipeline.encode(&mut compute_pass, texture);
    }
}