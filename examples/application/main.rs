use std::sync::Arc;
use pollster::FutureExt;
use wgpu::{Backends, CommandEncoderDescriptor, CompositeAlphaMode, CurrentSurfaceTexture, Device, DeviceDescriptor, ExperimentalFeatures, Extent3d, Features, FilterMode, Instance, InstanceDescriptor, InstanceFlags, Limits, MemoryHints, PollType, PowerPreference, PresentMode, Queue, RequestAdapterOptions, SubmissionIndex, Surface, SurfaceColorSpace, SurfaceConfiguration, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView, Trace, util::{TextureBlitter, TextureBlitterBuilder}};
use winit::{application::ApplicationHandler, event::WindowEvent, event_loop::{ActiveEventLoop, ControlFlow, EventLoop}, window::{Window, WindowId}};

enum RenderResult {
    Success,
    Retry,
    Skip,
    Lost
}

fn main() {
    let event_loop = match EventLoop::new() {
        Ok(event_loop) => event_loop,
        Err(err) => {
            return eprintln!("Failed to create event loop: {err:?}");
        }
    };

    event_loop.set_control_flow(ControlFlow::Wait);
    if let Err(err) = event_loop.run_app(&mut Application {
        cached_window: None,
        render_state: None
    }) {
        eprintln!("Failed to run event loop: {err:?}")
    }
}

struct Application {
    render_state: Option<RenderState>,
    cached_window: Option<Arc<Window>>
}

struct RenderState {
    window: Arc<Window>,
    surface: Surface<'static>,
    instance: Instance,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    texture_blitter: TextureBlitter,
    texture: Option<TextureView>,
    index: Option<SubmissionIndex>
}

impl ApplicationHandler for Application {
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = self.render_state.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(..) => state.window.request_redraw(),
            WindowEvent::RedrawRequested => loop {
                match state.render() {
                    RenderResult::Success => break,
                    RenderResult::Retry => continue,
                    RenderResult::Skip => break,
                    RenderResult::Lost => {
                        self.suspended(event_loop);
                        self.resumed(event_loop);
                        break;
                    }
                }
            }
            _ => ()
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.render_state.is_some() { return }
        let window = self.cached_window.take();
        self.render_state = RenderState::new(event_loop, window); 
    }

    fn suspended(&mut self, _: &ActiveEventLoop) {
        if let Some(state) = self.render_state.take() {
            self.cached_window = Some(state.window)
        }
    }
}

impl RenderState {
    fn new(event_loop: &ActiveEventLoop, window: Option<Arc<Window>>) -> Option<RenderState> {
        let window = match window {
            Some(window) => window,
            None => {
                let attributes = Window::default_attributes();
                match event_loop.create_window(attributes) {
                    Ok(window) => Arc::new(window),
                    Err(err) => {
                        eprintln!("Failed to create window: {err:?}");
                        return None;
                    }
                }
            }
        };

        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::PRIMARY,
            flags: InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER,
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: Some(Box::new(event_loop.owned_display_handle()))
        });

        let surface = match instance.create_surface(window.clone()) {
            Ok(surface) => surface,
            Err(err) => {
                eprintln!("Failed to create surface: {err:?}");
                return None;
            }
        };

        let adapter = match instance.request_adapter(&RequestAdapterOptions {
            compatible_surface: Some(&surface),
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            apply_limit_buckets: false
        }).block_on() {
            Ok(adapter) => adapter,
            Err(err) => {
                eprintln!("Failed to request adapter: {err:?}");
                return None;
            }
        };

        let features = adapter.features() & (
            Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES |
            Features::BGRA8UNORM_STORAGE
        );

        let (device, queue) = match adapter.request_device(&DeviceDescriptor {
            label: Some("Render Device"),
            required_features: features,
            experimental_features: ExperimentalFeatures::disabled(),
            required_limits: Limits::defaults(),
            memory_hints: MemoryHints::MemoryUsage,
            trace: Trace::Off
        }).block_on() {
            Ok(result) => result,
            Err(err) => {
                eprintln!("Failed to request device: {err:?}");
                return None;
            }
        }; 

        let size = Self::window_size(&window);
        let mut config = SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: TextureFormat::Bgra8Unorm,
            present_mode: PresentMode::AutoNoVsync,
            alpha_mode: CompositeAlphaMode::Auto,
            color_space: SurfaceColorSpace::Auto,
            desired_maximum_frame_latency: 1,
            view_formats: Vec::new(),
            height: size.height,
            width: size.width
        };

        let bgra8 = adapter.get_texture_format_features(TextureFormat::Bgra8Unorm);
        let capabilities = surface.get_capabilities(&adapter);
        let rgba8 = capabilities.formats.contains(&TextureFormat::Rgba8Unorm);
        let storage = capabilities.usages.contains(TextureUsages::STORAGE_BINDING);

        if rgba8 && storage {
            config.format = TextureFormat::Rgba8Unorm;
            config.usage |= TextureUsages::STORAGE_BINDING;
        } else if features.contains(Features::BGRA8UNORM_STORAGE) && storage {
            config.format = TextureFormat::Bgra8Unorm;
            config.usage |= TextureUsages::STORAGE_BINDING;
        } else if features.contains(Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) &&
            bgra8.allowed_usages.contains(TextureUsages::STORAGE_BINDING) && storage {
            config.format = TextureFormat::Bgra8Unorm;
            config.usage |= TextureUsages::STORAGE_BINDING;
        }

        let texture_blitter = TextureBlitterBuilder::new(&device, config.format)
            .sample_type(FilterMode::Nearest)
            .build();

        surface.configure(&device, &config);

        Some(RenderState {
            texture: None,
            index: None,
            texture_blitter,
            instance,
            config,
            device,
            queue,
            window,
            surface
        })
    }

    fn render(&mut self) -> RenderResult {
        let size = Self::window_size(&self.window);
        if size.width == 0 || size.height == 0 { return RenderResult::Skip }
        if size.width != self.config.width || size.height != self.config.height {
            self.config.width = size.width;
            self.config.height = size.height;
            self.surface.configure(&self.device, &self.config);
        }

        let output = match self.surface.get_current_texture() {
            CurrentSurfaceTexture::Timeout  |
            CurrentSurfaceTexture::Occluded |
            CurrentSurfaceTexture::Validation => return RenderResult::Skip,
            CurrentSurfaceTexture::Success(texture) => texture,
            CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return RenderResult::Retry;
            },
            CurrentSurfaceTexture::Suboptimal(output) => {
                drop(output);
                self.surface.configure(&self.device, &self.config);
                return RenderResult::Retry;
            },
            CurrentSurfaceTexture::Lost => {
                match self.instance.create_surface(self.window.clone()) {
                    Ok(surface) => {
                        self.surface = surface;
                        self.surface.configure(&self.device, &self.config);
                        return RenderResult::Retry
                    },
                    Err(..) => return RenderResult::Lost
                }
            }
        };

        let mut texture_view = output.texture.create_view(&Default::default());
        let mut target = None;

        if !self.config.usage.contains(TextureUsages::STORAGE_BINDING) {
            target = Some(texture_view);
            texture_view = self.texture.take()
            .filter(|view| view.texture().size() == size)
            .unwrap_or_else(|| self.device.create_texture(&TextureDescriptor {
                label: Some("Canvas Texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[]
            }).create_view(&Default::default()));
        }

        let mut command_encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Canvas Command Encoder")
        });

        if let Some(target) = target {
            self.texture_blitter.copy(
                &self.device,
                &mut command_encoder,
                &texture_view,
                &target
            );
        }

        let command_buffer = command_encoder.finish();
        let index = self.queue.submit(Some(command_buffer));
        self.queue.present(output);

        self.device.poll(PollType::Wait {
            submission_index: self.index.replace(index),
            timeout: None
        }).ok();

        RenderResult::Success
    }

    fn window_size(window: &Window)  -> Extent3d {
        let size = window.inner_size();
        Extent3d {
            width: size.width,
            height: size.height,
            depth_or_array_layers: 1
        }
    }
}