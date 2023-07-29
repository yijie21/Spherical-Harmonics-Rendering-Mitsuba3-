import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
from mitsuba import ScalarTransform4f as T
import numpy as np
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sphericalHarmonics import shTerms, shIndex, SH

def mis_weight(pdf_a, pdf_b):
    pdf_a *= pdf_a
    pdf_b *= pdf_b
    w = pdf_a / (pdf_a + pdf_b)
    return dr.detach(dr.select(dr.isfinite(w), w, 0))

def sph_to_dir(theta, phi):
    """Map spherical to Euclidean coordinates"""
    st, ct = dr.sincos(theta)
    sp, cp = dr.sincos(phi)
    return mi.Vector3f(cp * st, sp * st, ct)

def save_ckpt_np(model, outpath):
    for el in model.named_parameters():
        param_name = el[0]   # either fc1.bias or fc1.weight
        weights = el[1]
        segs = param_name.split('.')
        if segs[-1] == 'weight':
            param_name = segs[0]
        else:
            param_name = segs[0].replace('fc', 'b')

        filename = '_{}.npy'.format(param_name)
        filepath = os.path.join(outpath, filename)
        curr_weight = weights.detach().cpu().numpy().T  # transpose bc mitsuba code was developed for TF convention
        np.save(filepath, curr_weight)
      
def first_non_specular_or_null_si(scene, si, sampler):
    """Find the first non-specular or null surface interaction.

    Args:
        scene (mi.Scene): Scene object.
        si (mi.SurfaceInteraction3f): Surface interaction.
        sampler (mi.Sampler): Sampler object.

    Returns:
        tuple: A tuple containing four values:
            - si (mi.SurfaceInteraction3f): First non-specular or null surface interaction.
            - β (mi.Spectrum): The product of the weights of all previous BSDFs.
            - null_face (bool): A boolean mask indicating whether the surface is a null face or not.
    """
    # Instead of `bsdf.flags()`, based on `bsdf_sample.sampled_type`.
    with dr.suspend_grad():
        bsdf_ctx = mi.BSDFContext()

        depth = mi.UInt32(0)
        β = mi.Spectrum(1)
        bsdf = si.bsdf()

        null_face = ~mi.has_flag(si.bsdf().flags(), mi.BSDFFlags.BackSide) & (
            si.wi.z < 0
        )
        active = si.is_valid() & ~null_face  # non-null surface
        active &= ~mi.has_flag(si.bsdf().flags(), mi.BSDFFlags.Smooth)  # Delta surface

        loop = mi.Loop(
            name="first_non_specular_or_null_si",
            state=lambda: (sampler, depth, β, active, null_face, si, bsdf),
        )
        max_depth = 6
        loop.set_max_iterations(max_depth)

        while loop(active):
            # loop invariant: si is located at non-null and Delta surface
            # if si is located at null or Smooth surface, end loop
            bsdf_sample, bsdf_weight = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active
            )
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            si = scene.ray_intersect(
                ray, ray_flags=mi.RayFlags.All, coherent=dr.eq(depth, 0)
            )
            bsdf = si.bsdf(ray)

            β *= bsdf_weight
            depth[si.is_valid()] += 1

            null_face &= ~mi.has_flag(bsdf.flags(), mi.BSDFFlags.BackSide) & (
                si.wi.z < 0
            )
            active &= si.is_valid() & ~null_face & (depth < max_depth)
            active &= ~mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

    # return si at the first non-specular bounce or null face
    return si, β, null_face

def sample_sphere_dirs_world(res):    
    sampler = mi.load_dict({'type': 'independent', 'sample_count': res})
    sampler.seed(0, res)
    a = sampler.next_1d()
    b = sampler.next_1d()
    theta = 2 * dr.acos(dr.sqrt(1 - a))
    phi = 2 * dr.pi * b
    
    w = sph_to_dir(theta, phi)
    return w

def sample_sphere_dirs_grid(res):
    theta, phi = dr.meshgrid(
        dr.linspace(mi.Float, 0,     dr.pi,     res),
        dr.linspace(mi.Float, 0, 2 * dr.pi, 2 * res)
    )
    
    w = sph_to_dir(theta, phi)
    return w

def random_sample_light(n_samples, film_size, scene, sampler):
    sampler.seed(0, film_size)
    ray_random = mi.Ray3f(o=mi.Point3f(1, 0, 1), d=mi.Vector3f(0, 5, 1))
    si = scene.ray_intersect(ray_random, ray_flags=mi.RayFlags.All, coherent=True)
    emitter = scene.emitters()[0]
    l_dict = {}
    for i in range(n_samples):
        ds, em_weight = emitter.sample_direction(si, sampler.next_2d())
        em_val = scene.eval_emitter_direction(si, ds)
        d = {'ds': ds, 'em_weight': em_weight, 'em_val': em_val}
        l_dict[i] = d
    
    return l_dict
    
def get_visibility(scene, si, wi):
    rays = si.spawn_ray(wi)
    si_s = scene.ray_intersect(rays)
    valid = dr.eq(si_s.shape, None) | dr.neq(si_s.shape, si.shape)
    return valid

def get_visibility_global(scene, si, wi):
    rays = si.spawn_ray(wi)
    si_s = scene.ray_intersect(rays)
    valid = dr.eq(si_s.shape, None) | si_s.shape.is_emitter()
    return valid

def srgb_gamma(pixels):
    pixels = torch.pow(pixels, 1.0 / 2.2)
    return pixels

def sample_light_mitsuba(scene, n_samples):
    emitter = scene.emitters()[0]
    
    sampler = scene.sensors()[0].sampler().clone()
    sampler.seed(0, n_samples)
    random_ray = mi.Ray3f(o=mi.Vector3f(0, 0, 0), d=mi.Vector3f(1, 10, 1))
    random_si = scene.ray_intersect(random_ray)
    ds, _ = emitter.sample_direction(random_si, sampler.next_2d(), True)
    
    norm_d = dr.normalize(ds.d)
    theta = dr.acos(norm_d.z)
    phi = dr.atan2(norm_d.y, norm_d.x)
    
    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wi = -ds.d
    emitter_val = emitter.eval(si)
    emitter_pdf = emitter.pdf_direction(si, ds)
    emitter_weight = emitter_val / emitter_pdf
    
    return {
        'd': ds.d,
        'emitter_weight': emitter_weight,
        'emitter_pdf': emitter_pdf,
        'emitter_val': emitter_val,
        'theta': theta,
        'phi': phi
    }
    
def sample_light(scene, res, mode='random'):
    emitter = scene.emitters()[0]
    if mode == 'random':
        wi = sample_sphere_dirs_world(res)
    else:
        wi = sample_sphere_dirs_grid(res)
    
    ds = dr.zeros(mi.DirectionSample3f)
    ds.d = wi
    si = dr.zeros(mi.SurfaceInteraction3f)
    si.wi = -ds.d
    
    norm_d = dr.normalize(ds.d)
    theta = dr.acos(norm_d.z)
    phi = dr.atan2(norm_d.y, norm_d.x)
    
    emitter_val = emitter.eval(si)
    emitter_pdf = 1 / (4 * dr.pi)
    
    emitter_weight = emitter_val / emitter_pdf
    
    return {
        'd': ds.d,
        'emitter_weight': emitter_weight,
        'emitter_val': emitter_val,
        'theta': theta,
        'phi': phi
    }
    
def psnr(image1, image2):
    mse = F.mse_loss(image1, image2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()

def sensor_rays(scene, spp):
    sensor = scene.sensors()[0]
    sampler = scene.sensors()[0].sampler()
    sampler_spp = sampler.sample_count()
    sampler.set_sample_count(spp)
    film = sensor.film()
    
    film_size = film.crop_size()
    idx = dr.arange(mi.UInt32, dr.prod(film_size))
    idx = dr.tile(idx, spp)
    
    pos = mi.Vector2i()
    pos.y = idx // film_size[0]
    pos.x = dr.fma(-film_size[0], pos.y, idx)
    pos_f = mi.Vector2f(pos)# + sampler.next_2d()
    
    scale = dr.rcp(mi.ScalarVector2f(film.crop_size()))
    offset = -mi.ScalarVector2f(film.crop_offset()) * scale
    pos_adjusted = dr.fma(pos_f, scale, offset)
    
    aperture_sample = mi.Vector2f(0.0)
    
    ray, weight = sensor.sample_ray_differential(
        time=0,
        sample1=0,
        sample2=pos_adjusted,
        sample3=aperture_sample
    )
    
    sampler.set_sample_count(sampler_spp)
    
    return ray

def wo2wr(cam_ray, geo_n):
    wo = -cam_ray
    wr = 2 * (wo * geo_n) * geo_n - wo
    return -wr

def render_cameras(radius, theta_max, phi_max, n_frame):
    min_r = radius * 0.6
    max_r = radius * 1.4
    
    rs = torch.cat([torch.linspace(max_r, min_r, n_frame // 2),
                   torch.linspace(min_r, max_r, n_frame // 2)[1:]])
    thetas = torch.cat([torch.linspace(-theta_max, theta_max, n_frame // 2),
                        torch.linspace(theta_max, -theta_max, n_frame // 2)[1:]])
    phis = torch.cat([torch.linspace(-phi_max, phi_max, n_frame // 2),
                      torch.linspace(phi_max, -phi_max, n_frame // 2)[1:]])
    return torch.stack([rs, thetas, phis], dim=1)

def rotation_matrix(a, b):
    b = b / torch.norm(b, dim=1, keepdim=True) # normalize a
    a = a / torch.norm(a, dim=1, keepdim=True) # normalize b
    v = torch.cross(a, b)
    c = torch.sum(a * b, dim=1, keepdim=True)

    v1, v2, v3 = v[:, 0], v[:, 1], v[:, 2]
    h = 1 / (1 + c)
    
    zeros = torch.zeros_like(v1).to(a.device)
    
    row1 = torch.stack([zeros, -v3, v2], dim=1)
    row2 = torch.stack([v3, zeros, -v1], dim=1)
    row3 = torch.stack([-v2, v1, zeros], dim=1)
    Vmat = torch.stack([row1, row2, row3], dim=2)

    R = torch.eye(3)[None].repeat(a.shape[0], 1, 1).to(a.device) + Vmat + \
        (torch.bmm(Vmat, Vmat) * h[..., None])
    return R

def scene_bbox(scene):
    scene_bbox_min = []
    scene_bbox_max = []
    for shape in scene.shapes():
        if dr.eq(shape.is_emitter(), False):
            bbox_min = shape.bbox().min.torch()
            bbox_max = shape.bbox().max.torch()
            scene_bbox_min.append(bbox_min)
            scene_bbox_max.append(bbox_max)
    scene_bbox_min = torch.stack(scene_bbox_min, dim=0)
    scene_bbox_max = torch.stack(scene_bbox_max, dim=0)
    
    p_min = torch.Tensor([[scene_bbox_min[:, 0].min(), scene_bbox_min[:, 1].min(), scene_bbox_min[:, 2].min()]]).cuda()
    p_max = torch.Tensor([[scene_bbox_max[:, 0].max(), scene_bbox_max[:, 1].max(), scene_bbox_max[:, 2].max()]]).cuda()
    
    p_min = p_min - 0.01 * (p_max - p_min)
    p_max = p_max + 0.01 * (p_max - p_min)
    
    return p_min, p_max
        
def get_emitter_coe(scene, light_samples, l_band):
    nCoeffs = shTerms(l_band)
    l_dict = sample_light(scene, light_samples, 'random')
    theta_t = l_dict['theta'].torch()
    phi_t = l_dict['phi'].torch()
    emitter_val_t = l_dict['emitter_val'].torch()

    sh_basis_matrix = torch.zeros((light_samples, nCoeffs)).to(emitter_val_t.device)
    for l in range(l_band + 1):
        for m in range(-l, l + 1):
            index = shIndex(l, m)
            sh_basis_matrix[:, index] = SH(l, m, theta_t, phi_t)

    emitter_coe = torch.zeros((nCoeffs,3)).to(emitter_val_t.device)
    for i in range(nCoeffs):
        emitter_coe[i, 0] = torch.sum(emitter_val_t[:, 0] * sh_basis_matrix[:, i])
        emitter_coe[i, 1] = torch.sum(emitter_val_t[:, 1] * sh_basis_matrix[:, i])
        emitter_coe[i, 2] = torch.sum(emitter_val_t[:, 2] * sh_basis_matrix[:, i])
    emitter_coe = emitter_coe * 4 * torch.pi / light_samples
    return emitter_coe

def denoised_render(scene, spp):
    integrator = mi.load_dict({
        'type': 'aov',
        'aovs': 'albedo:albedo,normals:sh_normal',
        'integrator': {
            'type': 'path',
            'max_depth': 17
        }
    })

    sensor = scene.sensors()[0]
    to_sensor = sensor.world_transform().inverse()

    mi.render(scene, spp=spp, integrator=integrator)
    noisy_multichannel = sensor.film().bitmap()
    denoiser = mi.OptixDenoiser(input_size=noisy_multichannel.size(), albedo=True, normals=True, temporal=False)
    denoised = denoiser(noisy_multichannel, albedo_ch="albedo", normals_ch="normals", to_sensor=to_sensor)

    return denoised


def area_list(scene):
    m_area = []
    for shape in scene.shapes():
        m_area.append(shape.surface_area())
    m_area = np.array(m_area)[:, 0]
    m_area /= m_area.sum()
    return m_area


def sample_si(scene, shape_sampler, sample1, sample2, sample3, active=True):
    """Sample a batch of surface interactions with bsdfs.

    Args:
        scene (mitsuba.Scene): the underlying scene
        shape_sampler (mi.DiscreteDistribution): a source of random numbers for shape sampling
        sample1 (drjit.llvm.ad.Float): determines mesh surfaces
        sample2 (mitsuba.Point2f): determines positions on the meshes
        sample3 (mitsuba.Point2f): determines directions at the positions
        active (bool, optional): mask to specify active lanes. Defaults to True.

    Returns:
        mitsuba.SurfaceInteraction3f
    """
    shape_indices = shape_sampler.sample(sample1, active)
    shapes = dr.gather(mi.ShapePtr, scene.shapes_dr(), shape_indices, active)

    ps = shapes.sample_position(0, sample2, active)
    si = mi.SurfaceInteraction3f(ps, dr.zeros(mi.Color0f))
    si.shape = shapes
    si.t = mi.Float(si.time*0)

    # active_two_sided = mi.has_flag(si.bsdf().flags(), mi.BSDFFlags.BackSide)
    active_two_sided = False                # default False
    si.wi = dr.select(
        active_two_sided,
        mi.warp.square_to_uniform_sphere(sample3),
        mi.warp.square_to_uniform_hemisphere(sample3),
    )

    return si


def sample_si_reTrace(scene, shape_sampler, sample1, sample2, sample3, active=True):
    shape_indices = shape_sampler.sample(sample1, active)
    shapes = dr.gather(mi.ShapePtr, scene.shapes_dr(), shape_indices, active)
    
    ps = shapes.sample_position(0, sample2, active)

    # fetch_idx = []
    # for i, shape in enumerate(shapes):
        # if shape.id() == 'CookerBlack_0007':
            # id = shape_indices[i]
            # fetch_idx.append(i)
    
    # import open3d as o3d
    # p = ps.p.torch().cpu().numpy()
    # p_filter = p[fetch_idx]
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(p_filter)
    # o3d.io.write_point_cloud('pcd.ply', pcd)
    
    # si_ref
    si_ref = mi.SurfaceInteraction3f(ps, dr.zeros(mi.Color0f))
    si_ref.shape = shapes
    si_ref.t = mi.Float(si_ref.time*0)
    si_ref.wi = mi.warp.square_to_uniform_hemisphere(sample3)
    
    eps = 1e-4
    p = ps.p
    wo_world = si_ref.to_world(si_ref.wi)
    
    o = p + eps * wo_world
    d = -wo_world
    
    ray = mi.Ray3f(o=o, d=d)
    si = scene.ray_intersect(ray)
    
    return si, ray


def get_sampler(spp, wavefront_size, seed):
    sampler = mi.load_dict({'type': 'independent'})
    sampler.set_sample_count(spp)
    sampler.set_samples_per_wavefront(spp)
    sampler.seed(seed, wavefront_size)
    return sampler


def render_ray(scene, ray, spp, max_depth):
    # return not srgb_gamma correction torch lengthy pixels
    
    def sample_bsdf(sampler, active, bsdf_ctx, si, bsdf, active_next):

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

    def emitter_hit(scene, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si):

        # Compute MIS weight for emitter sample from previous bounce
        ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

        mis = mis_weight(
            prev_bsdf_pdf,
            scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
        )

        em_hit_result = throughput * mis * ds.emitter.eval(si)
        return em_hit_result

    def sample_emitter(scene, sampler, throughput, bsdf_ctx, si, bsdf, active_next):
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
    
    rr_depth = 5
    rr_prob = 0.95
    depth = mi.UInt32(0)
    eta = mi.Float(1)
    result = mi.Spectrum(0)
    throughput = mi.Spectrum(1)
    valid_ray = mi.Mask(dr.neq(scene.environment(), None))
    # active = mi.Bool(True)                      # Active SIMD lanes
    active = mi.Mask(True)
    
    wavefront_size = ray.o.torch().shape[0]
    sampler = get_sampler(spp, wavefront_size, 0)
    
    # Variables caching information from the previous bounce
    prev_si = dr.zeros(mi.SurfaceInteraction3f)
    prev_bsdf_pdf = mi.Float(1.0)
    prev_bsdf_delta = mi.Mask(True)
    bsdf_ctx = mi.BSDFContext()
    # Record the following loop in its entirety
    
    loop = mi.Loop(name="MyPathTracer",
                   state=lambda: (sampler, ray, throughput, result, valid_ray,
                                  eta, depth, prev_si, prev_bsdf_pdf,
                                  prev_bsdf_delta, active))
    # Specify the max. number of loop iterations (this can help avoid
    # costly synchronization when when wavefront-style loops are generated)
    loop.set_max_iterations(max_depth)
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
        em_hit_result = emitter_hit(
            scene, throughput, prev_si, prev_bsdf_pdf, prev_bsdf_delta, si)
        result += em_hit_result
        
        # ---------------------- Emitter sampling ----------------------
        # Should we continue tracing to reach one more vertex?
        active_next = (depth + 1 < max_depth) & si.is_valid()
        
        em_sample_result = sample_emitter(
            scene, sampler, throughput, bsdf_ctx, si, bsdf, active_next)
        result += em_sample_result
        
        # ------------------ Detached BSDF sampling -------------------
        bsdf_sample, bsdf_weight, ray = sample_bsdf(
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
        rr_prob = dr.minimum(throughput_max * eta**2, rr_prob)
        rr_active = depth >= rr_depth
        rr_continue = sampler.next_1d() < rr_prob
        throughput[rr_active] *= dr.rcp(dr.detach(rr_prob))
        active = active_next & (~rr_active | rr_continue) & dr.neq(throughput_max, 0)
    
    result = dr.select(valid_ray, result, 0)
    result_t = result.torch()
    result_t = torch.mean(result_t.reshape(spp, wavefront_size // spp, 3), dim=0)
    
    return result_t    


# drawing
# helper functions for visualization
def world_to_ndc(scene, batch):
    """Transforms 3D world coordinates into normalized device coordinates (NDC) using the perspective transformation matrix.

    Args:
        scene (mi.Scene): Mitsuba 3 scene containing the camera information.
        batch (array_like): Array of 3D world coordinates.

    Returns:
        mi.Point3f: Array of 3D points in NDC.
    """
    sensor = mi.traverse(scene.sensors()[0])
    trafo = mi.Transform4f.perspective(fov=sensor['x_fov'], near=sensor['near_clip'], far=sensor['far_clip'])
    pts = trafo @ sensor['to_world'].inverse() @ mi.Point3f(np.array(batch))
    return pts

def ndc_to_pixel(pts, h, w):
    """Converts points in NDC to pixel coordinates.

    Args:
        pts (mi.Point2f): Points in NDC.
        h (float): Height of the image in pixels.
        w (float): Width of the image in pixels.

    Returns:
        mi.Point2f: Pixel coordinates of the given points.
    """
    hh, hw = h/2, w/2
    return mi.Point2f(dr.fma(pts.x, -hw, hw), dr.fma(pts.y, -hw, hh))  # not typo

def draw_multi_segments(starts, ends, color):
    """Draws multiple line segments on a plot.

    Args:
        starts (mi.Point2f): Starting points of the line segments.
        ends (mi.Point2f): Ending points of the line segments.
        color (str): Color of the line segments.
    """
    a = np.c_[starts.x, starts.y]
    b = np.c_[ends.x, ends.y]
    plt.plot(*np.c_[a, b, a*np.nan].reshape(-1, 2).T, color)

def pix_coord(scene, batch, h, w):
    """Calculates the pixel coordinates of the given 3D world coordinates.

    Args:
        scene (mi.Scene): Mitsuba 3 scene containing the camera information.
        batch (array_like): Array of 3D world coordinates.
        h (float): Height of the image in pixels.
        w (float): Width of the image in pixels.

    Returns:
        mi.Point2f: Pixel coordinates of the given 3D world coordinates.
    """
    return ndc_to_pixel(world_to_ndc(scene, batch), h, w)

def draw_si(scene, si, original_image, marker=".", color="red"):
    """Draws the surface interaction on a plot.

    Args:
        si (mi.SurfaceInteraction): Surface interaction to be drawn.
        ax (matplotlib.axes.Axes): The axes object to plot the surface interaction on.
        original_image (np.ndarray): The original image to plot on the axes.
        marker (str): Marker style for the plot.
        color (str): Color for the plot.
    """
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)  # Hide the figure's background
    ax.axis('off')  # Remove the axes from the image
    fig.tight_layout()  # Remove any extra white spaces around the image
    
    h, w = original_image.shape[0], original_image.shape[1] 
    x_pix = pix_coord(scene, si.p, h, w)
    wi_pix = pix_coord(scene, dr.fma(si.to_world(si.wi), 0.25, si.p), h, w)
    n_pix = pix_coord(scene, dr.fma(si.n, 0.25, si.p), h, w)

    # draw directions
    draw_multi_segments(x_pix, n_pix, 'green')
    draw_multi_segments(x_pix, wi_pix, 'magenta')

    ax.scatter(x_pix.x, x_pix.y, marker=marker, color=color)
    plt.scatter(n_pix.x, n_pix.y, marker='.', color='green')
    plt.scatter(wi_pix.x, wi_pix.y, marker='x', color='magenta')
    ax.imshow(np.clip(original_image ** (1.0 / 2.2), 0, 1))

def draw_si_wi(scene, si, wi, original_image, marker=".", color="red"):
    """Draws the surface interaction on a plot.

    Args:
        si (mi.SurfaceInteraction): Surface interaction to be drawn.
        ax (matplotlib.axes.Axes): The axes object to plot the surface interaction on.
        original_image (np.ndarray): The original image to plot on the axes.
        marker (str): Marker style for the plot.
        color (str): Color for the plot.
    """
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)  # Hide the figure's background
    ax.axis('off')  # Remove the axes from the image
    fig.tight_layout()  # Remove any extra white spaces around the image
    
    h, w = original_image.shape[0], original_image.shape[1] 
    x_pix = pix_coord(scene, si.p, h, w)
    wo_pix = pix_coord(scene, dr.fma(si.to_world(si.wi), 0.25, si.p), h, w)
    wi_pix = pix_coord(scene, dr.fma(wi, 0.25, si.p), h, w)

    # draw directions
    draw_multi_segments(x_pix, wi_pix, 'green')
    draw_multi_segments(x_pix, wo_pix, 'magenta')

    ax.scatter(x_pix.x, x_pix.y, marker=marker, color=color)
    plt.scatter(wi_pix.x, wi_pix.y, marker='.', color='green')
    plt.scatter(wo_pix.x, wo_pix.y, marker='x', color='magenta')
    ax.imshow(np.clip(original_image ** (1.0 / 2.2), 0, 1))

def draw_moving_arrow(scene, si, h, w, si_prev, ax):
    """Draws a moving arrow on a plot to represent the surface interaction movement.

    Args:
        si (mi.SurfaceInteraction): Surface interaction to be drawn.
        h (float): Height of the image in pixels.
        w (float): Width of the image in pixels.
        si_prev (mi.SurfaceInteraction): Previous surface interaction to connect to the current one.
        ax (matplotlib.axes.Axes): The axes object to plot the moving arrow on.
    """
    x_pix = pix_coord(scene, si.p, h, w)
    x_prev_pix = pix_coord(scene, si_prev.p, h, w)
    draw_multi_segments(x_prev_pix, x_pix, 'green')