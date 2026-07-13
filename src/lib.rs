mod tile;
mod sort;
mod index;
mod alloc;
mod tessellation;

use std::num::NonZero;

use wgpu::{BindingResource, CommandEncoder, ComputePassDescriptor, Device, TextureFormat, TextureView};

use crate::alloc::Allocator;
use crate::index::{IndexBuffers, IndexPipeline};
use crate::sort::{SortBuffers, SortPipeline};
use crate::tessellation::{Path, Point, TessellationBuffers, TessellationPipeline, Triangle};
use crate::tile::{TileBuffers, TilePipeline};

pub struct DrawContext {
    tessellation_pipeline: TessellationPipeline,
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
            tessellation_pipeline: TessellationPipeline::new(device.clone()),
            allocator: Allocator::new(device.clone()),
            tile_pipeline: TilePipeline::new(device.clone(), format),
            sort_pipeline: SortPipeline::new(device.clone()),
            index_pipeline: IndexPipeline::new(device.clone())
        }
    }

    pub fn render(&mut self, encoder: &mut CommandEncoder, texture: TextureView, device: &Device, queue: &wgpu::Queue) {

        let size = texture.texture().size();
        let extent = size.width.max(size.height);
        let sample_fraction_bits = 15u32.checked_sub(u32::BITS - extent.leading_zeros())
            .filter(|x| *x >= 2).expect("Texture target too large");
        let list_count = Self::total_list_count(sample_fraction_bits);

        let paths = self.allocator.alloc::<Path>(1);
        let points = self.allocator.alloc::<Point>(3);
        let triangles = self.allocator.alloc::<Triangle>(points.len());
        let triangle_list_indices = self.allocator.alloc::<u32>(triangles.len());
        let triangle_indices = self.allocator.alloc::<u32>(triangles.len());
        
        let temp_keys = self.allocator.alloc::<u32>(triangles.len());
        let temp_lists = self.allocator.alloc::<u32>(triangles.len());
        let histogram_capacity = SortPipeline::min_histogram_capacity(triangles.len());
        let histograms = self.allocator.alloc::<[u32; 16]>(histogram_capacity);

        let list_ranges = self.allocator.alloc::<[u32; 2]>(list_count as usize);
        let params = self.allocator.alloc::<u32>(1);

        let mut memory = self.allocator.finalize();

        memory.upload(encoder, params, &[sample_fraction_bits]);
        memory.upload(encoder, paths, &[Path {
            fraction_bits: 0,
            length: 3,
            material: !0,
            offset: 0
        }]);
        memory.upload(encoder, points, &[
            Point { path: 0, value: [100, 100], padding: 0 },
            Point { path: 0, value: [200, 100], padding: 0 },
            Point { path: 0, value: [150, 200], padding: 0 }
        ]);

        if let Some(BindingResource::Buffer(list_ranges)) = memory.binding(list_ranges) {
            let size = list_ranges.size.map(NonZero::<u64>::get);
            encoder.clear_buffer(&list_ranges.buffer, list_ranges.offset, size);
        }

        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Canvas Compute Pass"),
            timestamp_writes: None
        });

        self.tessellation_pipeline.encode(&mut compute_pass, &mut memory, TessellationBuffers {
            paths,
            points,
            triangle_list_indices,
            triangles,
            uniforms: params,
            triangle_indices
        });

        self.sort_pipeline.encode(&mut compute_pass, &mut memory, SortBuffers {
            keys: triangle_list_indices,
            temp_keys,
            values: triangle_indices,
            temp_values: temp_lists,
            histograms,
        });

        self.index_pipeline.encode(&mut compute_pass, &mut memory, IndexBuffers {
            sorted_list_keys: triangle_list_indices,
            list_ranges
        });

        /*if let Some(binding) = memory.binding(list_ranges) {
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
        }*/

        self.tile_pipeline.encode(&mut compute_pass, &mut memory, TileBuffers {
            triangles,
            list_ranges,
            texture,
            lists: triangle_indices,
            params
        });
    }

    pub fn total_list_count(sample_fraction_bits: u32) -> u32 {
        let f = sample_fraction_bits.clamp(1, 11);
        let root_level = 11 - f;

        if root_level == 0 { return 1; }

        let bottom_base = Self::level_offset(root_level);
        let bottom_lists = 1u32 << (2 * root_level);
        bottom_base + bottom_lists
    }

    fn level_offset(level: u32) -> u32 {
        let alternating = 0x5555_5555u32;
        let low_bits = (1u32 << (2 * level)) - 1;
        let geometric = (alternating & low_bits) - 1;

        let square_sum = geometric << 2;
        let linear_sum = (1u32 << (level + 4)) - 32;
        let constant_sum = 16 * (level - 1);

        1 + square_sum + linear_sum + constant_sum
    }
}
