use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferUsages, ComputePass, ComputePipeline, ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages};

pub(super) struct SortPipeline {
    count_keys: ComputePipeline,
    count_histograms: ComputePipeline,
    uniform_offsets: Vec<u32>,
    uniforms: Buffer,
    device: Device,
}

impl SortPipeline {
    pub(super) fn new(device: Device) -> SortPipeline {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Sort Shader"),
            source: ShaderSource::Wgsl(include_str!("shader.wgsl").into())
        });

        let bind_group_layout_entires = [0, 1, 2, 3].map(|i| {
            BindGroupLayoutEntry {
                binding: i,
                count: None,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: if i == 3 {
                        BufferBindingType::Uniform
                    } else {
                        BufferBindingType::Storage { read_only: false }
                    },
                    has_dynamic_offset: i == 3,
                    min_binding_size: None
                }
            }
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Sort Bind Group Layout"),
            entries: bind_group_layout_entires.as_slice()
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Sort Pipeline Layout"),
            immediate_size: 0,
            bind_group_layouts: &[Some(&bind_group_layout)]
        });

        let count_keys = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Count Keys Pipeline"),
            cache: None,
            compilation_options: Default::default(),
            entry_point: Some("countKeys"),
            layout: Some(&pipeline_layout),
            module: &module
        });

        let count_histograms = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Count Histograms Pipeline"),
            cache: None,
            compilation_options: Default::default(),
            entry_point: Some("countHistograms"),
            layout: Some(&pipeline_layout),
            module: &module
        });

        let mut uniform_offsets = Vec::new();
        let mut uniform_contents = Vec::new();
        let alignment = device.limits().min_uniform_buffer_offset_alignment;

        for i in (0..u32::BITS).step_by(4) {
            let len = uniform_contents.len();
            let offset = len.next_multiple_of(alignment as usize);
            uniform_contents.resize(offset, 0);
            uniform_offsets.push(offset as u32);
            uniform_contents.extend_from_slice(&i.to_ne_bytes());
        }

        let uniforms = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: &uniform_contents ,
            usage: BufferUsages::UNIFORM
        });

        SortPipeline {
            count_histograms,
            uniform_offsets,
            count_keys,
            uniforms,
            device
        }
    }

    fn encode(&self, compute_pass: &mut ComputePass) {
        for offset in self.uniform_offsets.iter().copied() {

        }
    }
}