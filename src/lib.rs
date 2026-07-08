mod tile;
mod sort;

use wgpu::{CommandEncoder, ComputePassDescriptor, Device, TextureFormat, TextureView};

use crate::tile::TilePipeline;

pub struct DrawContext {
    device: Device,
    tile_pipeline: TilePipeline,
}

pub struct DrawList {

}


impl DrawContext {
    pub fn new(device: Device, format: TextureFormat) -> DrawContext {
        DrawContext {
            tile_pipeline: TilePipeline::new(device.clone(), format),
            device
        }
    }

    pub fn render(&mut self, encoder: &mut CommandEncoder, texture: TextureView) {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Canvas Compute Pass"),
            timestamp_writes: None
        });

        self.tile_pipeline.encode(&mut compute_pass, texture);
    }
}