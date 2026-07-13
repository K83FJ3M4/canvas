use bytemuck::{Pod, Zeroable};
use wgpu::{BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType, ComputePass, ComputePipeline, ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages};

use crate::alloc::{Allocation, AllocationMemory};

pub(super) struct TessellationPipeline {
    compute_pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    device: Device
}

#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy)]
pub(super) struct Point {
    pub(super) value: [i32; 2],
    pub(super) path: u32,
    pub(super) padding: u32
}

#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy)]
pub(super) struct Path {
    pub(super) fraction_bits: u32,
    pub(super) material: u32,
    pub(super) offset: u32,
    pub(super) length: u32
}

#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy)]
pub(super) struct Triangle {
    clockwise: u32,
    material: u32,
    a: [u32; 2],
    b: [u32; 2],
    c: [u32; 2],
}

pub(super) struct TessellationBuffers {
    pub(super) points: Allocation<Point>,
    pub(super) paths: Allocation<Path>,
    pub(super) triangle_list_indices: Allocation<u32>,
    pub(super) triangle_indices: Allocation<u32>,
    pub(super) triangles: Allocation<Triangle>,
    pub(super) uniforms: Allocation<u32>
}

impl TessellationPipeline {
    pub(super) fn new(device: Device) -> TessellationPipeline {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Tessellation Shader"),
            source: ShaderSource::Wgsl(include_str!("shader.wgsl").into())
        });

        let bind_group_layout_entires = [0, 1, 2, 3, 4, 5].map(|i| BindGroupLayoutEntry {
            binding: i,
            count: None,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None
            }
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Tessellation Bind Group Layout"),
            entries: &bind_group_layout_entires
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Tessellation Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0
        });

        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Tessellation Pipeline"),
            module: &module,
            entry_point: Some("main"),
            layout: Some(&pipeline_layout),
            compilation_options: Default::default(),
            cache: None
        });

        TessellationPipeline {
            bind_group_layout,
            compute_pipeline,
            device
        }
    }

    pub(super) fn encode(&self, compute_pass: &mut ComputePass, memory: &mut AllocationMemory, buffers: TessellationBuffers) {
        assert_eq!(buffers.triangles.len(), buffers.triangle_list_indices.len());
        assert_eq!(buffers.points.len(), buffers.triangles.len());

        let Some(points) = memory.binding(buffers.points) else { return };
        let Some(paths) = memory.binding(buffers.paths) else { return };
        let Some(triangle_list_indices) = memory.binding(buffers.triangle_list_indices) else { return };
        let Some(triangle_indices) = memory.binding(buffers.triangle_indices) else { return };
        let Some(triangles) = memory.binding(buffers.triangles) else { return };
        let Some(uniforms) = memory.binding(buffers.uniforms) else { return };

        let mut binding = 0;
        let bind_group_entries = [
            points, paths, triangle_list_indices,
            triangle_indices, triangles, uniforms
            ].map(|resource| {
            let entry = BindGroupEntry { binding, resource };
            binding += 1;
            entry
        });

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Tessellation Bind Group"),
            layout: &self.bind_group_layout,
            entries: &bind_group_entries
        });

        let dispatch_size = buffers.points.len().div_ceil(256) as u32;
        compute_pass.set_bind_group(0, &bind_group, Default::default());
        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
    }
}