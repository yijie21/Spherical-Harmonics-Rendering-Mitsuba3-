import gc
from typing import Tuple, Union

import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import torch
from tqdm import tqdm
from utils import mis_weight
import numpy as np
from utils import *
from collections import defaultdict
from typing import Any


class CounterProfiler:
    def __init__(self) -> None:
        self.data: dict[str, list[list[Any]]] = defaultdict(list)
        self.group_counter = 1
        self.enabled = False

    def record(self, name: str, value: Any):
        if not self.enabled:
            return

        gp_list = self.data[name]
        while len(gp_list) < self.group_counter:
            gp_list.append([])
        gp_list[-1].append(value)

    def new_group(self):
        if not self.enabled:
            return

        self.group_counter += 1


counter_profiler = CounterProfiler()


class HighQuality(mi.SamplingIntegrator):
    """
    This class takes care of rendering an image, but instead of rendering the whole pixels all at once,
    just rendering small blocks at a time and iteratively repeate the same for all blocks
    until the whole image is rendererd.
    
    Integrator: mi.SamplingIntegrator
        This is the integrator that takes care of rendering the blocks.
    """

    def __init__(self, block_size: int, integrator: mi.SamplingIntegrator):
        super().__init__(mi.Properties())
        self.block_size = block_size
        self.integrator = integrator

    def prepare(self,
                sensor: mi.Sensor,
                block: mi.ImageBlock,
                seed: int = 0,
                spp: int = 0,
                ):
        """
        This method is another implementation of method pepare in mitsuba/src/python/python/ad/common.py
        The difference is that it prepares the sampler to sample for a wavefront size that matches the block size not the film size.
        """

        film_on_sensor = sensor.film()
        sampler = sensor.sampler()

        if spp != 0:
            sampler.set_sample_count(spp)

        spp = sampler.sample_count()
        sampler.set_samples_per_wavefront(spp)

        block_size = block.size()

        if film_on_sensor.sample_border():
            raise NotImplementedError()

        wavefront_size = dr.prod(block_size) * spp

        if wavefront_size > 2**32:
            raise Exception(
                "The total number of Monte Carlo samples required by this "
                "rendering task (%i) exceeds 2^32 = 4294967296. Please use "
                "fewer samples per pixel or render using multiple passes."
                % wavefront_size)

        sampler.seed(seed, wavefront_size)

        return sampler, spp

    def render(self,
               scene: mi.Scene,
               sensor: Union[int, mi.Sensor] = 0,
               seed: int = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True) -> mi.TensorXf:
        """
        This method is another implementation of method render() in mitsuba/src/python/python/ad/common.py
        The difference is that it breaks down the rendering task into smaller comutations of blocks in the image instead of rendering it all at one pass.
        It uses the underlying integrator to render each block.
        """

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        # Prepare the spiral
        spiral = mi.Spiral(sensor.film().crop_size(), mi.ScalarVector2i(0, 0), self.block_size)
        sensor.film().prepare(self.integrator.aov_names())
        has_aov = len(self.integrator.aov_names()) > 0

        for i in tqdm(range(spiral.block_count())):
            block_offset, block_size, block_id = spiral.next_block()
            # Prepare an ImageBlock as specified by the film and block size
            block = sensor.film().create_block(block_size)
            block.set_offset(block_offset)

            # Disable derivatives in all of the following
            with dr.suspend_grad():
                with torch.no_grad():
                    # Prepare the film and sample generator for rendering
                    sampler, spp = self.prepare(
                        sensor=sensor,
                        block=block,
                        seed=(seed+1)*block_id,
                        spp=spp)
                    # Generate a set of rays starting at the sensor
                    ray, weight, pos = self.sample_rays(scene, sensor, sampler, block)
                    # Launch the Monte Carlo sampling process in primal mode
                    if issubclass(type(self.integrator), mi.CppADIntegrator):
                        L, valid, _ = self.integrator.sample(
                            mode=dr.ADMode.Primal,
                            scene=scene,
                            sampler=sampler,
                            ray=ray,
                            depth=mi.UInt32(0),
                            δL=None,
                            state_in=None,
                            reparam=None,
                            active=mi.Bool(True)
                        )
                    else:
                        L, valid, aov = self.integrator.sample(
                            scene,
                            sampler,
                            ray,
                            None,
                            active=mi.Bool(True))

                    # Only use the coalescing feature when rendering enough samples
                    # block.set_coalesce(block.coalesce() and spp >= 4)

                    # Accumulate into the image block
                    alpha = dr.select(valid, mi.Float(1), mi.Float(0))
                    if has_aov:
                        # Assumption: weight is always [1.0, 1.0, 1.0]
                        floatLs = [L[0], L[1], L[2], alpha, weight[0]]
                        all_channels = floatLs + aov
                        block.put(pos, all_channels)
                        del aov
                    else:
                        block.put(pos, ray.wavelengths, L * weight, alpha)

                    sampler.schedule_state()
                    dr.eval(block.tensor())

                    # Explicitly delete any remaining unused variables
                    del ray, weight, pos, L, valid, alpha
                    gc.collect()

                    # Perform the weight division and return an image tensor
                    sensor.film().put_block(block)
                    torch.cuda.empty_cache()
                    dr.flush_malloc_cache()

        primal_image = sensor.film().develop()
        dr.schedule(primal_image)
        if evaluate:
            dr.eval()
            dr.sync_thread()

        return primal_image

    def render_forward(self: mi.Integrator,
                       scene: mi.Scene,
                       params,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: int = 0,
                       spp: int = 0) -> mi.TensorXf:
        """
        This method is another implementation of method render() in mitsuba/src/python/python/ad/common.py
        The difference is that it breaks down the rendering task into smaller comutations of blocks in the image instead of rendering it all at one pass.
        It uses the underlying integrator to render each block.
        """

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        # Prepare the spiral
        spiral = mi.Spiral(sensor.film().crop_size(), mi.ScalarVector2i(0, 0), self.block_size)
        sensor.film().prepare(self.integrator.aov_names())
        has_aov = len(self.integrator.aov_names()) > 0

        for i in tqdm(range(spiral.block_count())):
            block_offset, block_size, block_id = spiral.next_block()
            # Prepare an ImageBlock as specified by the film and block size
            block = sensor.film().create_block(block_size)
            block.set_offset(block_offset)

            # Disable derivatives in all of the following
            with dr.suspend_grad():

                # Prepare the film and sample generator for rendering
                sampler, spp = self.prepare(
                    sensor=sensor,
                    block=block,
                    seed=(seed+1)*block_id,
                    spp=spp)
                # Generate a set of rays starting at the sensor
                ray, weight, pos = self.sample_rays(scene, sensor, sampler, block)
                # Launch the Monte Carlo sampling process in primal mode

                with dr.resume_grad():
                    dr.enable_grad(params)
                    dr.set_grad(params, 1)

                    if issubclass(type(self.integrator), mi.ad.common.ADIntegrator):
                        L, valid, _ = self.integrator.sample(
                            mode=dr.ADMode.Primal,
                            scene=scene,
                            sampler=sampler,
                            ray=ray,
                            depth=mi.UInt32(0),
                            δL=None,
                            state_in=None,
                            reparam=None,
                            active=mi.Bool(True)
                        )
                    else:
                        L, valid, aov = self.integrator.sample(
                            scene,
                            sampler,
                            ray,
                            None,
                            active=mi.Bool(True))

                    dr.forward_to(L)
                    L = dr.grad(L)
                    dr.disable_grad(params)

                # Only use the coalescing feature when rendering enough samples
                # block.set_coalesce(block.coalesce() and spp >= 4)

                # Accumulate into the image block
                alpha = dr.select(valid, mi.Float(1), mi.Float(0))
                if has_aov:
                    # Assumption: weight is always [1.0, 1.0, 1.0]
                    floatLs = [L[0], L[1], L[2], alpha, weight[0]]
                    all_channels = floatLs + aov
                    block.put(pos, all_channels)
                    del aov
                else:
                    block.put(pos, ray.wavelengths, L * weight, alpha)

                sampler.schedule_state()
                dr.eval(block.tensor())

                # Explicitly delete any remaining unused variables
                del ray, weight, pos, L, valid, alpha
                gc.collect()

                # Perform the weight division and return an image tensor
                sensor.film().put_block(block)
                torch.cuda.empty_cache()
                dr.flush_malloc_cache()

        grad_image = sensor.film().develop()
        dr.schedule(grad_image)

        return grad_image

    def sample_rays(
        self,
        scene: mi.Scene,
        sensor: mi.Sensor,
        sampler: mi.Sampler,
        block: mi.ImageBlock
    ) -> Tuple[mi.RayDifferential3f, mi.Spectrum, mi.Vector2f, mi.Float]:
        """
        This method is another implementation of method sample_rays() in mitsuba/src/python/python/ad/common.py
        The difference is that it prepares the samples for a block of rays instead of the whole image plane pixels.
        """

        block_size = block.size()
        rfilter = sensor.film().rfilter()
        border_size = rfilter.border_size()

        if sensor.film().sample_border():
            block_size += 2 * border_size
            raise NotImplementedError()

        spp = sampler.sample_count()

        # Compute discrete sample position
        idx = dr.arange(mi.UInt32, dr.prod(block_size) * spp)

        # Try to avoid a division by an unknown constant if we can help it
        log_spp = dr.log2i(spp)
        if 1 << log_spp == spp:
            idx >>= dr.opaque(mi.UInt32, log_spp)
        else:
            idx //= dr.opaque(mi.UInt32, spp)

        # Compute the position on the image plane
        pos = mi.Vector2i()
        pos.y = idx // block_size[0]
        pos.x = dr.fma(-block_size[0], pos.y, idx)

        if sensor.film().sample_border():
            pos -= border_size
            raise NotImplementedError()

        pos += mi.Vector2i(block.offset())

        # Cast to floating point and add random offset
        pos_f = mi.Vector2f(pos) + sampler.next_2d()

        # Re-scale the position to [0, 1]^2
        scale = dr.rcp(mi.ScalarVector2f(sensor.film().crop_size()))
        # offset = -mi.ScalarVector2f(block.offset()) * scale #TODO: check why this is wrong in the orignial mitsuba.ad.common.py
        pos_adjusted = pos_f * scale

        aperture_sample = mi.Vector2f(0.0)
        if sensor.needs_aperture_sample():
            aperture_sample = sampler.next_2d()

        time = sensor.shutter_open()
        if sensor.shutter_open_time() > 0:
            time += sampler.next_1d() * sensor.shutter_open_time()

        wavelength_sample = 0
        if mi.is_spectral:
            wavelength_sample = sampler.next_1d()

        with dr.resume_grad():
            ray, weight = sensor.sample_ray_differential(
                time=time,
                sample1=wavelength_sample,
                sample2=pos_adjusted,
                sample3=aperture_sample
            )

        # With box filter, ignore random offset to prevent numerical instabilities
        splatting_pos = mi.Vector2f(pos) if rfilter.is_box_filter() else pos_f
        return ray, weight, splatting_pos

    def to_string(self):
        return (
            "Highquality[\n"

            "]"
        )


class MyPathTracer(mi.SamplingIntegrator):
    """Path Tracer (AD-PT)

    Code is tranlated from Mitsuba C++.
    """

    def __init__(self):
        super().__init__(mi.Properties())
        self._init()

    def _init(
        self,
        hide_emitters=False,
        return_depth=False,
        max_depth=17,
        rr_depth=5,
        rr_prob: float = 0.95,
        **kwargs
    ):
        self.hide_emitters = hide_emitters
        self.return_depth = return_depth
        self.rr_prob = rr_prob

        # max depth
        if max_depth < 0 and max_depth != -1:
            raise Exception(
                "\"max_depth\" must be set to -1 (infinite) or a value >= 0")

        # Map -1 (infinity) to 2^32-1 bounces
        self.max_depth = max_depth if max_depth != -1 else 0xffffffff

        if rr_depth <= 0:
            raise Exception(
                "\"rr_depth\" must be set to a value greater than zero!")
        self.rr_depth = rr_depth

    def sample(self,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               medium: mi.Medium,
               active: mi.Bool):

        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)
        eta = mi.Float(1)
        result = mi.Spectrum(0)
        throughput = mi.Spectrum(1)
        valid_ray = mi.Mask((~mi.Bool(self.hide_emitters))
                            & dr.neq(scene.environment(), None))

        active = mi.Bool(active)                      # Active SIMD lanes

        # Variables caching information from the previous bounce
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        bsdf_ctx = mi.BSDFContext()

        # Record the following loop in its entirety
        loop = mi.Loop(name="MyPathTracer",
                       state=lambda: (sampler, ray, throughput, result,
                                      eta, depth, valid_ray, prev_si, prev_bsdf_pdf,
                                      prev_bsdf_delta, active))

        # Specify the max. number of loop iterations (this can help avoid
        # costly synchronization when when wavefront-style loops are generated)
        loop.set_max_iterations(self.max_depth)

        while loop(active):
            # Compute a surface interaction that tracks derivatives arising
            # from differentiable shape parameters (position, normals, etc.)
            # In primal mode, this is just an ordinary ray tracing operation.

            si = scene.ray_intersect(ray,
                                     ray_flags=mi.RayFlags.All,
                                     coherent=dr.eq(depth, 0))

            # Get the BSDF, potentially computes texture-space differentials
            bsdf = si.bsdf(ray)

            # ---------------------- Direct emission ----------------------

            em_hit_result = self.emitter_hit(
                scene, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si)
            result += em_hit_result
            # ---------------------- Emitter sampling ----------------------

            # Should we continue tracing to reach one more vertex?
            active_next = (depth + 1 < self.max_depth) & si.is_valid()

            em_sample_result = self.sample_emitter(
                scene, sampler, throughput, bsdf_ctx, si, bsdf, active_next)
            result += em_sample_result

            # ------------------ Detached BSDF sampling -------------------

            bsdf_sample, bsdf_weight, ray = self.bsdf_sample(
                sampler, active, bsdf_ctx, si, bsdf, active_next)

            # ------ Update loop variables based on current interaction ------

            throughput *= bsdf_weight
            eta *= bsdf_sample.eta
            valid_ray |= active & si.is_valid() & ~mi.has_flag(
                bsdf_sample.sampled_type, mi.BSDFFlags.Null)

            # Information about the current vertex needed by the next iteration
            prev_si = si
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(
                bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------

            depth[si.is_valid()] += 1
            # Don't run another iteration if the throughput has reached zero
            throughput_max = dr.max(throughput)
            rr_prob = dr.minimum(throughput_max * eta**2, self.rr_prob)
            rr_active = depth >= self.rr_depth
            rr_continue = sampler.next_1d() < rr_prob
            throughput[rr_active] *= dr.rcp(dr.detach(rr_prob))
            active = active_next & (
                ~rr_active | rr_continue) & dr.neq(throughput_max, 0)

        aov = [depth] if self.return_depth else []
        if counter_profiler.enabled:
            counter_profiler.record("integrator.depth", np.array(depth).tolist())

        return dr.select(valid_ray, result, 0), valid_ray, aov

    def aov_names(self):
        return ['depth'] if self.return_depth else []

    def bsdf_sample(self, sampler, active, bsdf_ctx, si, bsdf, active_next):

        bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                               sampler.next_1d(),
                                               sampler.next_2d(),
                                               active_next)

        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

        # When the path tracer is differentiated, we must be careful that
        #   the generated Monte Carlo samples are detached (i.e. don't track
        #   derivatives) to avoid bias resulting from the combination of moving
        #   samples and discontinuous visibility. We need to re-evaluate the
        #   BSDF differentiably with the detached sample in that case. */
        if (dr.grad_enabled(ray)):
            ray = dr.detach(ray)

            # Recompute 'wo' to propagate derivatives to cosine term
            wo = si.to_local(ray.d)
            bsdf_val, bsdf_pdf = bsdf.eval_pdf(bsdf_ctx, si, wo, active)
            bsdf_weight[bsdf_pdf > 0] = bsdf_val / dr.detach(bsdf_pdf)

        return bsdf_sample, bsdf_weight, ray

    def emitter_hit(self, scene, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si):

        # Compute MIS weight for emitter sample from previous bounce
        ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

        mis = mis_weight(
            prev_bsdf_pdf,
            scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
        )

        em_hit_result = throughput * mis * ds.emitter.eval(si)
        return em_hit_result

    def sample_emitter(self, scene, sampler, throughput, bsdf_ctx, si, bsdf, active_next):
        # Is emitter sampling even possible on the current vertex?
        active_em = active_next & mi.has_flag(
            bsdf.flags(), mi.BSDFFlags.Smooth)

        # If so, randomly sample an emitter without derivative tracking.
        ds, em_weight = scene.sample_emitter_direction(
            si, sampler.next_2d(), True, active_em)
        active_em &= dr.neq(ds.pdf, 0.0)

        if (dr.grad_enabled(si.p)):
            # Given the detached emitter sample, *recompute* its
            # contribution with AD to enable light source optimization
            ds.d = dr.normalize(ds.p - si.p)
            em_val = scene.eval_emitter_direction(si, ds, active_em)
            em_weight = dr.select(dr.neq(ds.pdf, 0), em_val / ds.pdf, 0)

            # Evaluate BSDF * cos(theta) differentiably
        wo = si.to_local(ds.d)
        bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
        mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
        em_sample_result = throughput * mis_em * bsdf_value_em * em_weight

        return em_sample_result

    def to_string(self):
        return (
            "MyPathTracer[\n"
            f"    max_depth={self.max_depth},\n"
            f"    rr_depth={self.rr_depth},\n"
            "]"
        )
