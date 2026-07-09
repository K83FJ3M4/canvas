use wgpu::{BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType, ComputePass, ComputePipeline, ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages};
use crate::alloc::{Allocation, AllocationMemory};

pub(super) struct IndexPipeline {
    compute_pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    device: Device
}

pub(super) struct IndexBuffers {
    pub(super) sorted_list_keys: Allocation<u32>,
    pub(super) list_ranges: Allocation<[u32; 2]>,
}

impl IndexPipeline {
    pub(super) fn new(device: Device) -> IndexPipeline {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Index Shader"),
            source: ShaderSource::Wgsl(include_str!("shader.wgsl").into())
        });

        let bind_group_layout_entires = [0, 1].map(|i| BindGroupLayoutEntry {
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
            label: Some("Index Bind Group Layout"),
            entries: &bind_group_layout_entires
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Index Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0
        });

        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Index Pipeline"),
            module: &module,
            entry_point: Some("main"),
            layout: Some(&pipeline_layout),
            compilation_options: Default::default(),
            cache: None
        });

        IndexPipeline {
            compute_pipeline,
            bind_group_layout,
            device
        }
    }

    pub(super) fn encode(&self, compute_pass: &mut ComputePass, memory: &mut AllocationMemory, buffers: IndexBuffers) {
        let Some(sorted_list_keys) = memory.binding(buffers.sorted_list_keys) else { return };
        let Some(list_ranges) = memory.binding(buffers.list_ranges) else { return };

        let mut binding = 0;
        let bind_group_entries = [sorted_list_keys, list_ranges].map(|resource| {
            let entry = BindGroupEntry { binding, resource };
            binding += 1;
            entry
        });

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Index Bind Group"),
            layout: &self.bind_group_layout,
            entries: &bind_group_entries
        });

        let dispatch_size = buffers.sorted_list_keys.len().div_ceil(256) as u32;
        compute_pass.set_bind_group(0, &bind_group, Default::default());
        compute_pass.set_pipeline(&self.compute_pipeline);
        compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
    }
}