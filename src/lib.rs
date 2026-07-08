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

    pub fn render(&mut self, encoder: &mut CommandEncoder, texture: TextureView) {

        let keys = self.allocator.alloc::<u32>(513);
        let temp_keys = self.allocator.alloc::<u32>(keys.len());
        let histograms = self.allocator.alloc::<[u32; 16]>(12);
        let mut memory = self.allocator.finalize();

        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Canvas Compute Pass"),
            timestamp_writes: None
        });

        self.sort_pipeline.encode(&mut compute_pass, &mut memory, SortBuffers {
            keys,
            temp_keys,
            histograms,
        });

        self.tile_pipeline.encode(&mut compute_pass, texture);
    }
}