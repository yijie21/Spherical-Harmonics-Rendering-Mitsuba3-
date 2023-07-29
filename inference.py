import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import torch
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from sphericalHarmonics import *
import einops
from mitsuba import ScalarTransform4f as T


def mannual_render(scene, ctx, si, bsdf, d, emitter_weight):
    wo_local = si.to_local(d)
    vis = get_visibility_global(scene, si, d)
    bsdf_val, _ = bsdf.eval_pdf(ctx, si, wo_local)
    bsdf_val = si.to_world_mueller(bsdf_val, -wo_local, si.wi)
    result = dr.zeros(mi.Spectrum)
    result[vis] += emitter_weight * bsdf_val
    
    return result.torch()
    
    
def sh_rendering(scene, ctx, si, bsdf, l_dict, n_samples, l_band):
    d = l_dict['d']
    emitter_val_t = l_dict['emitter_val'].torch()
    theta_t = l_dict['theta'].torch()
    phi_t = l_dict['phi'].torch()
    
    nCoeffs = shTerms(l_band)
    
    sh_basis_matrix = torch.zeros((n_samples, nCoeffs)).to(emitter_val_t.device)
    for l in range(l_band + 1):
        for m in range(-l, l + 1):
            index = shIndex(l, m)
            sh_basis_matrix[:, index] = SH(l, m, theta_t, phi_t)
    
    emitter_coe = torch.zeros((nCoeffs,3)).to(emitter_val_t.device)
    for i in range(nCoeffs):
        emitter_coe[i, 0] = torch.sum(emitter_val_t[:, 0] * sh_basis_matrix[:, i])
        emitter_coe[i, 1] = torch.sum(emitter_val_t[:, 1] * sh_basis_matrix[:, i])
        emitter_coe[i, 2] = torch.sum(emitter_val_t[:, 2] * sh_basis_matrix[:, i])
    emitter_coe = emitter_coe * 4 * torch.pi / n_samples
    
    im_size = scene.sensors()[0].film().crop_size()
    w = im_size[0]
    h = im_size[1]
    
    bsdfs_coe = torch.zeros((h * w, nCoeffs, 3)).cuda()
    for i in tqdm(range(n_samples)):
        d_i = dr.gather(type(d), d, i)
        wo_i = si.to_local(d_i)
        vis = get_visibility_global(scene, si, d_i) 
        bsdf_val, _ = bsdf.eval_pdf(ctx, si, wo_i)
        bsdf_val = si.to_world_mueller(bsdf_val, -wo_i, si.wi)
        bsdf_val_vis = dr.zeros(mi.Spectrum)
        bsdf_val_vis[vis] = bsdf_val
        
        bsdf_val_t = bsdf_val_vis.torch()
        bsdfs_coe += bsdf_val_t[:, None] * sh_basis_matrix[i:i+1, :, None]
    
    bsdfs_coe = bsdfs_coe * 4 * torch.pi / n_samples
    result = (bsdfs_coe * emitter_coe[None]).sum(dim=1)
    result = torch.abs(result)
    return result


def sh_rendering_mixed(scene, ctx, si, bsdf, l_dict_light, l_dict_bsdf, l_band):
    d = l_dict_bsdf['d']
    emitter_val_t = l_dict_light['emitter_val'].torch()
    theta_bsdf_t = l_dict_bsdf['theta'].torch()
    phi_bsdf_t = l_dict_bsdf['phi'].torch()
    theta_light_t = l_dict_light['theta'].torch()
    phi_light_t = l_dict_light['phi'].torch()
    
    nCoeffs = shTerms(l_band)
    im_size = scene.sensors()[0].film().crop_size()
    w = im_size[0]
    h = im_size[1]
    
    n_samples = l_dict_light['emitter_val'].torch().shape[0]
    sh_basis_matrix_light = torch.zeros((n_samples, nCoeffs)).to(emitter_val_t.device)
    for l in range(l_band + 1):
        for m in range(-l, l + 1):
            index = shIndex(l, m)
            sh_basis_matrix_light[:, index] = SH(l, m, theta_light_t, phi_light_t)
    
    emitter_coe = torch.zeros((nCoeffs,3)).to(emitter_val_t.device)
    for i in range(nCoeffs):
        emitter_coe[i, 0] = torch.sum(emitter_val_t[:, 0] * sh_basis_matrix_light[:, i])
        emitter_coe[i, 1] = torch.sum(emitter_val_t[:, 1] * sh_basis_matrix_light[:, i])
        emitter_coe[i, 2] = torch.sum(emitter_val_t[:, 2] * sh_basis_matrix_light[:, i])
    emitter_coe = emitter_coe * 4 * torch.pi / n_samples
    
    
    n_samples = l_dict_bsdf['emitter_val'].torch().shape[0]
    sh_basis_matrix_bsdf = torch.zeros((n_samples, nCoeffs)).to(emitter_val_t.device)
    for l in range(l_band + 1):
        for m in range(-l, l + 1):
            index = shIndex(l, m)
            sh_basis_matrix_bsdf[:, index] = SH(l, m, theta_bsdf_t, phi_bsdf_t)
    bsdfs_coe = torch.zeros((h * w, nCoeffs, 3)).cuda()
    for i in tqdm(range(n_samples)):
        d_i = dr.gather(type(d), d, i)
        wo_i = si.to_local(d_i)
        vis = get_visibility_global(scene, si, d_i) 
        bsdf_val, _ = bsdf.eval_pdf(ctx, si, wo_i)
        bsdf_val = si.to_world_mueller(bsdf_val, -wo_i, si.wi)
        bsdf_val_vis = dr.zeros(mi.Spectrum)
        bsdf_val_vis[vis] = bsdf_val
        
        bsdf_val_t = bsdf_val_vis.torch()
        bsdfs_coe += bsdf_val_t[:, None] * sh_basis_matrix_bsdf[i:i+1, :, None]
    
    bsdfs_coe = bsdfs_coe * 4 * torch.pi / n_samples
    
    result = (bsdfs_coe * emitter_coe[None]).sum(dim=1)
    result = torch.abs(result)
    return result
   
   
def sh_rendering_grid(scene, ctx, si, bsdf, l_dict, res, l_band):
    d = l_dict['d']
    emitter_val_t = l_dict['emitter_val'].torch()
    n_samples = emitter_val_t.shape[0]
    emitter_val_t = emitter_val_t.reshape((res * 2, res, 3)).permute(1, 0, 2)
    emitter_coe = getCoefficientsFromMap(emitter_val_t, l_band)
    
    im_size = scene.sensors()[0].film().crop_size()
    w = im_size[0]
    h = im_size[1]
    
    nCoeffs = shTerms(l_band)
    sh_basis_matrix = getCoefficientsMatrix(res * 2, l_band + 1).cuda()
    solidAngles = getSolidAngleMap(res * 2).cuda()
    
    sh_basis_matrix = sh_basis_matrix.permute(1, 0, 2).reshape(-1, nCoeffs)
    solidAngles = solidAngles.permute(1, 0).reshape(-1)
    
    bsdfs_coe = torch.zeros((h * w, nCoeffs, 3)).cuda()
    for i in tqdm(range(n_samples)):
        d_i = dr.gather(type(d), d, i)
        wo_i = si.to_local(d_i)
        vis = get_visibility(scene, si, d_i)
        bsdf_val, _ = bsdf.eval_pdf(ctx, si, wo_i)
        bsdf_val = si.to_world_mueller(bsdf_val, -wo_i, si.wi)
        bsdf_val_vis = dr.zeros(mi.Spectrum)
        bsdf_val_vis[vis] = bsdf_val
        bsdf_val_t = bsdf_val_vis.torch()
        bsdfs_coe += bsdf_val_t[:, None] * sh_basis_matrix[i:i+1, :, None] * solidAngles[i]
    
    result = (bsdfs_coe * emitter_coe[None]).sum(dim=1)
    result = torch.abs(result)
    return result
   
   
def sh_rendering_mitsuba_smaple(scene, ctx, si, bsdf, l_dict, n_samples, l_band):
    d = l_dict['d']
    emitter_val_t = l_dict['emitter_val'].torch()
    emitter_pdf_t = l_dict['emitter_pdf'].torch()
    emitter_weight_t = l_dict['emitter_weight'].torch()
    theta_t = l_dict['theta'].torch()
    phi_t = l_dict['phi'].torch()
    
    nCoeffs = shTerms(l_band)
    
    sh_basis_matrix = torch.zeros((n_samples, nCoeffs)).to(emitter_val_t.device)
    for l in range(l_band + 1):
        for m in range(-l, l + 1):
            index = shIndex(l, m)
            sh_basis_matrix[:, index] = SH(l, m, theta_t, phi_t)
    
    emitter_coe = torch.zeros((nCoeffs,3)).to(emitter_val_t.device)
    for i in range(nCoeffs):
        emitter_coe[i, 0] = torch.sum(emitter_weight_t[:, 0] * sh_basis_matrix[:, i]) / n_samples
        emitter_coe[i, 1] = torch.sum(emitter_weight_t[:, 1] * sh_basis_matrix[:, i]) / n_samples
        emitter_coe[i, 2] = torch.sum(emitter_weight_t[:, 2] * sh_basis_matrix[:, i]) / n_samples
    
    im_size = scene.sensors()[0].film().crop_size()
    w = im_size[0]
    h = im_size[1]
    
    bsdfs_coe = torch.zeros((h * w, nCoeffs, 3)).cuda()
    for i in tqdm(range(n_samples)):
        d_i = dr.gather(type(d), d, i)
        wo_i = si.to_local(d_i)
        vis = get_visibility_global(scene, si, d_i) 
        bsdf_val, _ = bsdf.eval_pdf(ctx, si, wo_i)
        bsdf_val = si.to_world_mueller(bsdf_val, -wo_i, si.wi)
        bsdf_val_vis = dr.zeros(mi.Spectrum)
        bsdf_val_vis[vis] += bsdf_val
        
        bsdf_val_t = bsdf_val_vis.torch()
        bsdfs_coe += bsdf_val_t[:, None] * sh_basis_matrix[i:i+1, :, None] / emitter_pdf_t[i] / n_samples
    
    result = (bsdfs_coe * emitter_coe[None]).sum(dim=1)
    result = torch.abs(result)
    return result
   
   
def sh_rendering_diff_dirs(scene, ctx, si, bsdf, l_dict, n_samples, l_band):
    im_size = scene.sensors()[0].film().crop_size()
    w = im_size[0]
    h = im_size[1]
    
    sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(0, h * w)
    
    # TODO: currently, only one light source is supported
    emitter = scene.emitters()[0]
    
    # first: transfer coefficients
    nCoeffs = shTerms(l_band)
    
    transfer_coe = torch.zeros((h * w, nCoeffs, 3)).cuda()
    for i in tqdm(range(n_samples)):
        ds, emitter_weight = emitter.sample_direction(si, sampler.next_2d())
        emitter_pdf = emitter.pdf_direction(si, ds)
        wo = si.to_local(ds.d)
        vis = get_visibility_global(scene, si, ds.d)
        bsdf_val, _ = bsdf.eval_pdf(ctx, si, wo, True)
        bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)
        bsdf_val_vis = dr.zeros(mi.Spectrum)
        bsdf_val_vis[vis] += bsdf_val
        bsdf_val_vis_t = bsdf_val_vis.torch()
        
        norm_d = dr.normalize(ds.d)
        theta_t = dr.acos(norm_d.z).torch()
        phi_t = dr.atan2(norm_d.y, norm_d.x).torch()
        
        sh_basis = torch.zeros((h * w, nCoeffs)).cuda()
        for l in range(l_band + 1):
            for m in range(-l, l + 1):
                index = shIndex(l, m)
                sh_basis[:, index] = SH(l, m, theta_t, phi_t)
        transfer_coe += bsdf_val_vis_t[:, None] * sh_basis[:, :, None] / emitter_pdf.torch()[:, None, None] / n_samples
        
    # second: light coefficients
    emitter_val_t = l_dict['emitter_val'].torch()
    n_samples = emitter_val_t.shape[0]
    theta_t = l_dict['theta'].torch()
    phi_t = l_dict['phi'].torch()
    
    sh_basis_matrix = torch.zeros((n_samples, nCoeffs)).to(emitter_val_t.device)
    for l in range(l_band + 1):
        for m in range(-l, l + 1):
            index = shIndex(l, m)
            sh_basis_matrix[:, index] = SH(l, m, theta_t, phi_t)
    
    emitter_coe = torch.zeros((nCoeffs,3)).to(emitter_val_t.device)
    for i in range(nCoeffs):
        emitter_coe[i, 0] = torch.sum(emitter_val_t[:, 0] * sh_basis_matrix[:, i])
        emitter_coe[i, 1] = torch.sum(emitter_val_t[:, 1] * sh_basis_matrix[:, i])
        emitter_coe[i, 2] = torch.sum(emitter_val_t[:, 2] * sh_basis_matrix[:, i])
    emitter_coe = emitter_coe * 4 * torch.pi / n_samples
    
    result = (transfer_coe * emitter_coe[None]).sum(dim=1)
    result = torch.abs(result)
    return result


def sh_rendering_diff_dirs_spp(scene, ctx, si, bsdf, l_dict, bsdf_samples, batch_spp, l_band):
    im_size = scene.sensors()[0].film().crop_size()
    w = im_size[0]
    h = im_size[1]
    
    sampler = mi.load_dict({'type': 'independent'})
    spp = batch_spp
    sampler.set_sample_count(spp)
    sampler.set_samples_per_wavefront(spp)
    
    # TODO: currently, only one light source is supported
    emitter = scene.emitters()[0]
    
    # first: transfer coefficients
    nCoeffs = shTerms(l_band)
    
    transfer_coe = torch.zeros((h * w, nCoeffs, 3)).cuda()
    for i in tqdm(range(bsdf_samples // spp)):
        sampler_ = sampler.clone()
        sampler_.seed(i, h * w * spp)
        ds, _ = emitter.sample_direction(si, sampler_.next_2d())
        emitter_pdf = emitter.pdf_direction(si, ds)
        wo = si.to_local(ds.d)
        vis = get_visibility_global(scene, si, ds.d)
        bsdf_val, _ = bsdf.eval_pdf(ctx, si, wo, True)
        bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)
        bsdf_val_vis = dr.zeros(mi.Spectrum)
        bsdf_val_vis[vis] += bsdf_val
        bsdf_val_vis_t = bsdf_val_vis.torch()
        
        norm_d = dr.normalize(ds.d)
        theta_t = dr.acos(norm_d.z).torch()
        phi_t = dr.atan2(norm_d.y, norm_d.x).torch()
        
        sh_basis = torch.zeros((h * w * spp, nCoeffs)).cuda()
        for l in range(l_band + 1):
            for m in range(-l, l + 1):
                index = shIndex(l, m)
                sh_basis[:, index] = SH(l, m, theta_t, phi_t)
        
        tmp_coe = bsdf_val_vis_t[:, None] * sh_basis[:, :, None] / emitter_pdf.torch()[:, None, None] / bsdf_samples
        
        transfer_coe += torch.sum(einops.rearrange(tmp_coe, '(s l) h c -> s l h c', s=spp), dim=0)
    
    # second: light coefficients
    emitter_val_t = l_dict['emitter_val'].torch()
    light_samples = emitter_val_t.shape[0]
    theta_t = l_dict['theta'].torch()
    phi_t = l_dict['phi'].torch()
    
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
    
    result = (transfer_coe * emitter_coe[None]).sum(dim=1)
    result = torch.abs(result)
    return result

    
def neu_sh_rendering(scene, model, si, bsdf, light_samples, l_band, p_min, p_max):
    emitter_coe = get_emitter_coe(scene, light_samples, l_band)
    uv_t = si.uv.torch()
    wo_t = -si.to_world(si.wi).torch()
    wo_local_t = -si.wi.torch()
    geo_n_t = si.n.torch()
    p_t = si.p.torch()
    p_t = (p_t - p_min) / (p_max - p_min)
    albedo_t = bsdf.eval_diffuse_reflectance(si, si.is_valid()).torch()
    
    transfer_coe = model(p_t, wo_local_t, albedo_t)
    transfer_coe = transfer_coe.reshape(-1, emitter_coe.shape[0], 3)
    
    result = (transfer_coe * emitter_coe[None]).sum(dim=1)
    result = torch.abs(result)
    return result


# working functions
def render_mode(scene_name, mode='random'):
    xml_path = 'scenes/' + scene_name + '/scene.xml'
    scene = mi.load_file(xml_path)
    
    im_size = scene.sensors()[0].film().crop_size()
    w = im_size[0]
    h = im_size[1]

    batch_spp = 5
    light_samples = 5000
    bsdf_samples = 1000
    l_band = 4
    
    l_dict = sample_light(scene, light_samples, 'random')
    
    ray1 = sensor_rays(scene, 1)
    si1 = scene.ray_intersect(ray1)
    not_emitter = dr.eq(si1.shape.is_emitter(), False).torch() == 1
    result = si1.emitter(scene).eval(si1).torch()
    
    ray = sensor_rays(scene, batch_spp)
    si = scene.ray_intersect(ray)
    ctx = mi.BSDFContext()
    bsdf = si.bsdf(ray)
    
    if mode == 'sh':
        result_object = sh_rendering_diff_dirs_spp(scene, ctx, si, bsdf, l_dict, bsdf_samples, batch_spp, l_band)
        result[not_emitter] += result_object[not_emitter]
    else:
        for i in tqdm(range(bsdf_samples)):
            d_i = dr.gather(type(l_dict['d']), l_dict['d'], i)
            emitter_weight_i = dr.gather(type(l_dict['emitter_weight']), l_dict['emitter_weight'], i)
            result += mannual_render(scene, ctx, si, bsdf, d_i, emitter_weight_i) / bsdf_samples
        
    result = result.reshape((h, w, 3))
    result = np.clip(srgb_gamma(result).detach().cpu().numpy(), 0, 1)
    filname = scene_name + '_' + mode + '_' + str(bsdf_samples) + '.png'
    plt.imsave(filname, result)
    

if __name__ == '__main__':
    if len(sys.argv) < 3:
        scene_name = 'brass'
        mode = 'sh'
    else:
        scene_name = sys.argv[1]
        mode = sys.argv[2]
    render_mode(scene_name, mode)
    