-- https://github.com/ImagicTheCat/love-DPBR
-- MIT license (see LICENSE or love-DPBR.lua)

--[[
MIT License

Copyright (c) 2020 ImagicTheCat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
]]

local MATERIAL_SHADER = [[
#pragma language glsl3

uniform float scene_depth;
uniform Image MainTex;
uniform Image m_normal;
uniform Image m_MR;
uniform Image m_DE;
uniform float[2] m_depth_args; // (z, depth_max)
uniform float[2] m_emission_args; // factor, emission_max
uniform float[2] m_MR_args; // metalness and roughness factors
uniform int[2] m_color_profiles; // (albedo, MR) (0: linear, 1: sRGB)

float normalizeLog(float v, float max){ return log2(v+1.0)/log2(max+1.0); }
float denormalizeLog(float v, float max){ return pow(max, v)-1.0; }

mat3 getFragmentTBN()
{
  vec3 t = normalize(vec3(dFdx(VaryingTexCoord.xy),0.0));
  vec3 b = normalize(vec3(dFdy(VaryingTexCoord.xy),0.0));
  vec3 n = cross(t,b);
  return mat3(t,b,n);
}

void effect()
{
  // compute transformed normal
  vec4 n_color = Texel(m_normal, VaryingTexCoord.xy);
  vec3 n = getFragmentTBN()*(n_color.xyz*2.0-1.0); // compute derivatives before discard

  vec4 albedo = Texel(MainTex, VaryingTexCoord.xy)*VaryingColor;
  if(albedo.a == 0) discard; // discard transparent albedo
  if(m_color_profiles[0] == 1) // sRGB, transform to linear
    albedo.rgb = pow(albedo.rgb, vec3(2.2));

  vec4 MR = Texel(m_MR, VaryingTexCoord.xy);
  if(m_color_profiles[1] == 1) // sRGB, transform to linear
    MR.rgb = pow(MR.rgb, vec3(2.2));
  MR.rg *= vec2(m_MR_args[0], m_MR_args[1]); // factors

  vec4 DE = Texel(m_DE, VaryingTexCoord.xy); // depth + emission

  if(scene_depth > 0){ // orthographic mode
    if(m_depth_args[1] > 0) // denormalize depth
      gl_FragDepth = normalizeLog(denormalizeLog(DE.x, m_depth_args[1])+m_depth_args[0], scene_depth);
    else
      gl_FragDepth = normalizeLog(DE.x+m_depth_args[0], scene_depth);
  }
  else // custom mode
    gl_FragDepth = DE.x;

  love_Canvases[0] = albedo;
  love_Canvases[1] = vec4(n*0.5+0.5, albedo.a); // normal
  love_Canvases[2] = vec4(MR.rgb, albedo.a);

  if(m_emission_args[1] > 0) // normalized emission
    love_Canvases[3] = vec4(vec3(m_emission_args[0]*denormalizeLog(DE.y, m_emission_args[1])), albedo.a);
  else
    love_Canvases[3] = vec4(vec3(m_emission_args[0]*DE.y), albedo.a);
}
]]

local TRANSLUCENT_SHADER = [[
#pragma language glsl3

uniform float scene_depth;
uniform Image MainTex;
uniform Image m_DE;
uniform float[2] m_depth_args; // (z, depth_max)
uniform float[2] m_emission_args; // factor, emission_max
uniform int[2] m_color_profiles; // (albedo, MR) (0: linear, 1: sRGB)

float normalizeLog(float v, float max){ return log2(v+1.0)/log2(max+1.0); }
float denormalizeLog(float v, float max){ return pow(max, v)-1.0; }

void effect()
{
  vec4 albedo = Texel(MainTex, VaryingTexCoord.xy)*VaryingColor;
  if(albedo.a == 0) discard; // discard transparent albedo
  if(m_color_profiles[0] == 1) // sRGB, transform to linear
    albedo.rgb = pow(albedo.rgb, vec3(2.2));

  vec4 DE = Texel(m_DE, VaryingTexCoord.xy); // depth + emission

  if(scene_depth > 0){ // orthographic mode
    if(m_depth_args[1] > 0) // denormalize depth
      gl_FragDepth = normalizeLog(denormalizeLog(DE.x, m_depth_args[1])+m_depth_args[0], scene_depth);
    else
      gl_FragDepth = normalizeLog(DE.x+m_depth_args[0], scene_depth);
  }
  else // custom mode
    gl_FragDepth = DE.x;

  love_Canvases[0] = albedo;

  if(m_emission_args[1] > 0) // normalized emission
    love_Canvases[1] = vec4(vec3(m_emission_args[0]*denormalizeLog(DE.y, m_emission_args[1])), albedo.a);
  else
    love_Canvases[1] = vec4(vec3(m_emission_args[0]*DE.y), albedo.a);
}
]]


local LIGHTS = {
  AMBIENT = 0,
  POINT = 1,
  DIRECTIONAL = 2,
  EMISSION = 3,
  RAW = 4
}

local LIGHT_SHADER = [[
#pragma language glsl3
#define PI 3.1415926535897932384626433832795

uniform float scene_depth;
uniform mat4 projection_matrix;
uniform Image MainTex; // albedo
uniform Image g_depth;
uniform Image g_normal;
uniform Image g_MR;
uniform Image g_emission;
uniform int l_type;
uniform float l_intensity;
uniform vec4 l_pos_rad; // light position (view) and radius
uniform vec3 l_dir;

float denormalizeLog(float v, float max){ return pow(max, v)-1.0; }

// compute fragment's position, normal and view vector (fragment->view) in view space
void getFragmentPNV(out vec3 p, out vec3 n, out vec3 v)
{
  mat4 invp = inverse(projection_matrix);

  // position
  vec3 ndc = vec3(love_PixelCoord/love_ScreenSize.xy, 1.0);
  ndc.y = 1.0-ndc.y;
  ndc.z = Texel(g_depth, VaryingTexCoord.xy).x;
  if(scene_depth > 0) ndc.z = denormalizeLog(ndc.z, scene_depth)/scene_depth; // orthographic mode
  ndc = ndc*2.0-vec3(1.0);

  vec4 p4 = invp*vec4(ndc, 1.0);
  p4.z = -p4.z;
  p = vec3(p4/p4.w);

  // normal
  n = normalize(Texel(g_normal, VaryingTexCoord.xy).xyz*2.0-vec3(1.0))*vec3(1.0,-1.0,-1.0);

  // view
  vec4 vp = invp*vec4(ndc.xy, -1.0, 1.0);
  vp.z = -vp.z;
  vp /= vp.w;
  v = normalize(vec3(vp)-p);
}

// PBR functions
// N: normal vector
// L: light vector (fragment -> light)
// V: view vector (fragment -> view)
// H: halfway vector (normalized L+V)

vec3 fresnelSchlick(float HdotV, vec3 F0)
{
  return F0+(1.0-F0)*pow(1.0-HdotV, 5.0);
}

vec3 fresnel(float HdotV, vec3 albedo, float metalness)
{
  return fresnelSchlick(HdotV, mix(vec3(0.04), albedo, metalness));
}

// Trowbridge-Reitz GGX normal distribution function
float distributionGGX(float NdotH, float a)
{
  float a2 = a*a;
  float part = NdotH*NdotH*(a2-1.0)+1.0;
  return a2/(PI*part*part);
}

float geometrySchlickGGX(float NdotV, float k)
{
  return NdotV/(NdotV*(1.0-k)+k);
}

float geometrySmith(float NdotV, float NdotL, float k)
{
  return geometrySchlickGGX(NdotV, k)*geometrySchlickGGX(NdotL, k);
}

vec3 cookTorranceSpecular(float NdotH, float NdotV, float NdotL, vec3 fresnel, float a, float k)
{
  // tweak: prevent 0 division
  return distributionGGX(NdotH, a)*geometrySmith(NdotV, NdotL, k)*fresnel/max(4*NdotV*NdotL, 1e-3);
}

vec3 reflectance(vec3 radiance, float HdotV, float NdotH, float NdotV, float NdotL, vec3 albedo, float metalness, float a, float k)
{
  vec3 fresnel = fresnel(HdotV, albedo, metalness);
  vec3 kD = vec3(1.0)-fresnel; // kD - kS
  kD *= 1.0-metalness; // tweak

  // Lambertian Cook-Torrance BRDF
  return (kD*albedo/PI+cookTorranceSpecular(NdotH, NdotV, NdotL, fresnel, a, k))*radiance*NdotL;
}

// helper for point/dir. based lights
// l: light vector (fragment -> light)
vec3 rayLight(vec3 radiance, vec3 p, vec3 n, vec3 v, vec3 l)
{
  vec4 albedo = Texel(MainTex, VaryingTexCoord.xy);
  vec4 MR = Texel(g_MR, VaryingTexCoord.xy);

  vec3 h = normalize(l+v);
  float NdotH = max(dot(n,h), 0.0);
  float NdotV = max(dot(n,v), 0.0);
  float NdotL = max(dot(n,l), 0.0);
  float HdotV = max(dot(h,v), 0.0);
  float a = MR.y*MR.y;
  float k = (MR.y+1.0)*(MR.y+1.0)/8.0; // direct lighting

  return reflectance(radiance, HdotV, NdotH, NdotV, NdotL, albedo.rgb, MR.x, a, k);
}

void effect()
{
  if(l_type == 0){ // ambient
    vec4 albedo = Texel(MainTex, VaryingTexCoord.xy);
    love_Canvases[0] = vec4(albedo.rgb*l_intensity*VaryingColor.rgb, 1.0);
  }
  else if(l_type == 1){ // point
    vec3 p, n, v;
    getFragmentPNV(p,n,v);

    if(l_pos_rad.w > 0){
      vec3 pl = l_pos_rad.xyz-p;
      float distance = length(pl);
      float falloff_p1 = clamp(1.0-pow(distance/l_pos_rad.w, 4.0), 0.0, 1.0);
      float falloff = falloff_p1*falloff_p1/(distance*distance+1.0);
      love_Canvases[0] = vec4(rayLight(VaryingColor.rgb*l_intensity*falloff, p, n, v, normalize(pl)), 1.0);
    }
    else
      love_Canvases[0] = vec4(rayLight(VaryingColor.rgb*l_intensity, p, n, v, normalize(l_pos_rad.xyz-p)), 1.0);
  }
  else if(l_type == 2){ // directional
    vec3 p, n, v;
    getFragmentPNV(p,n,v);

    love_Canvases[0] = vec4(rayLight(VaryingColor.rgb*l_intensity, p, n, v, normalize(-l_dir)), 1.0);
  }
  else if(l_type == 3){ // emission
    vec4 albedo = Texel(MainTex, VaryingTexCoord.xy);
    float emission = Texel(g_emission, VaryingTexCoord.xy).x;
    love_Canvases[0] = vec4(albedo.rgb*emission*l_intensity*VaryingColor.rgb, 1.0);
  }
  else if (l_type == 4) // raw
    love_Canvases[0] = vec4(VaryingColor.rgb*Texel(MainTex, VaryingTexCoord.xy).rgb*l_intensity, 1.0);
}
]]

local TMO = {
  raw = 0,
  reinhard = 1,
  filmic = 2 -- Jim Hejl, Richard Burgess-Dawson
}

local BLOOM_EXTRACT_SHADER = [[
#pragma language glsl3
uniform vec2 texel_size;
uniform vec4 bloom_curve;
uniform float bloom_safe_clamp;

vec3 median(vec3 a, vec3 b, vec3 c){ return a+b+c-min(min(a, b), c)-max(max(a, b), c); }

vec3 sample(Image tex, vec2 uv)
{
  // clamp to prevent NaN (black artifacts)
  return clamp(Texel(tex, uv).rgb, vec3(0.0), vec3(bloom_safe_clamp));
}

vec4 effect(vec4 color, Image tex, vec2 uv, vec2 screen_coords)
{
  // 5-tap median filter (anti-flicker / smooth square artifacts)
  vec3 d = texel_size.xyx*vec3(1, 1, 0);
  vec3 s0 = sample(tex, uv.xy);
  vec3 s1 = sample(tex, uv.xy-d.xz);
  vec3 s2 = sample(tex, uv.xy+d.xz);
  vec3 s3 = sample(tex, uv.xy-d.zy);
  vec3 s4 = sample(tex, uv.xy+d.zy);
  vec3 hdr = median(median(s0, s1, s2), s3, s4);

  float brightness = max(hdr.r, max(hdr.g, hdr.b));
  float soft = clamp(brightness-bloom_curve.y, 0, bloom_curve.z);
  soft = soft*soft*bloom_curve.w;
  float contribution = max(soft, brightness-bloom_curve.x)/max(brightness, 1e-5);

  return vec4(hdr*contribution, 1.0);
}
]]

local BLOOM_DOWNSAMPLE_SHADER = [[
#pragma language glsl3
uniform vec2 texel_size;

vec4 effect(vec4 color, Image tex, vec2 uv, vec2 screen_coords)
{
  // 2x2 (4x4) blur filter
  vec4 d = texel_size.xyxy*vec4(-1,-1,1,1);
  vec3 m = Texel(tex, uv+d.xy).rgb;
  m += Texel(tex, uv+d.zy).rgb;
  m += Texel(tex, uv+d.xw).rgb;
  m += Texel(tex, uv+d.zw).rgb;

  return vec4(m/4.0, 1.0);
}
]]

local BLOOM_UPSAMPLE_SHADER = [[
#pragma language glsl3
uniform vec2 texel_size;
uniform float sample_scale;
uniform float intensity;

vec4 effect(vec4 color, Image tex, vec2 uv, vec2 screen_coords)
{
  // 2x2 (4x4) blur filter
  vec4 d = texel_size.xyxy*vec4(-1,-1,1,1)*0.5*sample_scale;

  vec3 m = Texel(tex, uv+d.xy).rgb;
  m += Texel(tex, uv+d.zy).rgb;
  m += Texel(tex, uv+d.xw).rgb;
  m += Texel(tex, uv+d.zw).rgb;

  return vec4(m/4.0*intensity, 1.0);
}
]]

local RENDER_SHADER = [[
#pragma language glsl3

uniform Image MainTex;
uniform float gamma;
uniform float exposure;
uniform int TMO;

void effect()
{
  vec4 hdra = Texel(MainTex, VaryingTexCoord.xy);

  // exposure adjustment
  vec3 hdr = hdra.rgb*exposure;

  // tone mapping
  if(TMO == 1) // reinhard
    hdr /= hdr+vec3(1.0);
  else if(TMO == 2){ // filmic
    hdr = max(vec3(0.0), hdr-vec3(0.004));
    hdr = (hdr*(6.2*hdr+.5))/(hdr*(6.2*hdr+1.7)+0.06);
  }

  // gamma correction
  if(TMO != 2) // bypass gamma correction (filmic)
    hdr = pow(hdr, vec3(1.0/gamma));

  // clamp
  hdr = clamp(hdr, vec3(0.0), vec3(1.0));

  love_Canvases[0] = vec4(hdr, hdra.a);
  // luma
  love_Canvases[1] = vec4(hdr.r*0.299+hdr.g*0.587+hdr.b*0.114);
}
]]

local FXAA_SHADER = [[
#pragma language glsl3

uniform Image g_luma;
uniform vec2 texel_size;
uniform float contrast_threshold;
uniform float relative_threshold;
uniform float subpixel_blending;

#define EDGE_STEP_COUNT 10
#define EDGE_STEPS 1, 1.5, 2, 2, 2, 2, 2, 2, 2, 4
#define EDGE_GUESS 8

const float[] edge_steps = float[](EDGE_STEPS);

// uv: current texel
// ln,lm,lp: luma negative/middle/positive
// step: (out) blend vector
// return edge blend factor
float computeEdgeBlending(const vec2 uv, bool horizontal, const float ln, const float lm, const float lp, out vec2 step)
{
  // gradients
  float gp = abs(lp-lm);
  float gn = abs(ln-lm);

  vec2 edge_step;
  if(horizontal){
    step = vec2(0.0, texel_size.y);
    edge_step = vec2(texel_size.x, 0.0);
  }
  else{
    step = vec2(texel_size.x, 0.0);
    edge_step = vec2(0.0, texel_size.y);
  }

  float threshold, l_edge;
  if(gp > gn){
    threshold = gp/4.0;
    l_edge = (lm+lp)/2.0;
  }
  else{
    step *= -1.0;
    threshold = gn/4.0;
    l_edge = (lm+ln)/2.0;
  }

  // positive walk
  vec2 start = uv+step*0.5;
  vec2 uvc = start;
  bool end = false;
  float pdelta;
  for(int i = 0; i < EDGE_STEP_COUNT && !end; i++){
    uvc += edge_step*edge_steps[i];
    pdelta = abs(Texel(g_luma, uvc).x-l_edge);
    end = (pdelta >= threshold);
  }
  if(!end) uvc += edge_step*EDGE_GUESS;
  float pdist = length(uvc-start);

  // negative walk
  uvc = start;
  end = false;
  float ndelta;
  for(int i = 0; i < EDGE_STEP_COUNT && !end; i++){
    uvc -= edge_step*edge_steps[i];
    ndelta = abs(Texel(g_luma, uvc).x-l_edge);
    end = (ndelta >= threshold);
  }
  if(!end) uvc -= edge_step*EDGE_GUESS;
  float ndist = length(uvc-start);

  float shortest, delta;
  if(pdist <= ndist){
    shortest = pdist;
    delta = pdelta;
  }
  else{
    shortest = ndist;
    delta = ndelta;
  }

  if(sign(delta) == sign(lm-l_edge)) // skip blending opposed deltas
    return 0.0;

  return 0.5-shortest/(pdist+ndist);
}

vec4 effect(vec4 vcolor, Image tex, vec2 uv, vec2 screen_coords)
{
  // luma MNSWE
  float lm = Texel(g_luma, uv).x;
  float ln = Texel(g_luma, uv+texel_size*vec2(0,1)).x;
  float ls = Texel(g_luma, uv+texel_size*vec2(0,-1)).x;
  float lw = Texel(g_luma, uv+texel_size*vec2(-1,0)).x;
  float le = Texel(g_luma, uv+texel_size*vec2(1,0)).x;

  float high = max(lm, max(ln, max(ls, max(lw, le))));
  float low = min(lm, min(ln, min(ls, min(lw, le))));
  float contrast = high-low;

  // skip pixels
  if(contrast < max(contrast_threshold, relative_threshold*high))
    return Texel(tex, uv);

  // compute subpixel blend factor

  // luma NE NW SE SW
  float lnw = Texel(g_luma, uv+texel_size*vec2(-1,1)).x;
  float lne = Texel(g_luma, uv+texel_size*vec2(1,1)).x;
  float lsw = Texel(g_luma, uv+texel_size*vec2(-1,-1)).x;
  float lse = Texel(g_luma, uv+texel_size*vec2(1,-1)).x;

  float blend_sp = (2.0*(ln+ls+lw+le)+lnw+lne+lsw+lse)/12.0;
  blend_sp = min(abs(blend_sp-lm)/contrast, 1.0);
  blend_sp = smoothstep(0.0, 1.0, blend_sp);
  blend_sp = blend_sp*blend_sp*subpixel_blending;

  // edge orientation detection
  float edge_h = 2.0*abs(ln+ls-2.0*lm)+abs(lne+lse-2.0*le)+abs(lnw+lsw-2.0*lw);
  float edge_v = 2.0*abs(lw+le-2.0*lm)+abs(lnw+lne-2.0*ln)+abs(lsw+lse-2.0*ls);

  float blend_edge;
  vec2 step;
  if(edge_h > edge_v) // horizontal edge
    blend_edge = computeEdgeBlending(uv, true, ls, lm, ln, step);
  else // vertical edge
    blend_edge = computeEdgeBlending(uv, false, le, lm, lw, step);

  return Texel(tex, uv+step*max(blend_sp, blend_edge));
}
]]

local DPBR = {}

local Scene = {}
local Scene_meta = {__index = Scene}

-- Create a scene.
--
-- A scene defines a 2D-3D space (view space), parameters and data to properly
-- render each material/object.
--
-- There are two modes for a scene: orthographic or custom.
-- The orthographic mode allows for any kind of 2D rendering, with the
-- possibility to adjust the depth of each element and perform meaningful
-- transformations. The depth is positive, view->far. Correct scene dimensions
-- are important to keep consistency for light computation (distance, etc.).
-- The custom mode allows the use of a specific projection matrix, but reduces
-- the API possibilities.
--
-- API requiring a position (like a point light) are in view space (defined by the projection).
--
-- w,h: render dimensions
-- depth_projection: max depth of the scene or custom projection matrix
--- max depth: orthographic mode, maximum depth distance of the scene (> 0)
--- projection matrix: custom mode, LÖVE table format (row-major)
-- sw, sh: (optional) scene dimensions for orthographic mode (default: w, h)
-- return Scene
function DPBR.newScene(w, h, depth_projection, sw, sh)
  local scene = setmetatable({}, Scene_meta)

  scene.w, scene.h = w, h
  scene:setProjection(depth_projection, sw, sh)

  -- init buffers
  scene.g_albedo = love.graphics.newCanvas(w, h, {format = "rgba8"})
  scene.g_normal = love.graphics.newCanvas(w, h, {format = "rgba8"})
  scene.g_MR = love.graphics.newCanvas(w, h, {format = "rg8"}) -- metalness + roughness
  scene.g_emission = love.graphics.newCanvas(w, h, {format = "r32f"})
  scene.g_depth = love.graphics.newCanvas(w, h, {format = "depth24", readable = true})
  scene.g_light = love.graphics.newCanvas(w, h, {format = "rgba32f"})
  scene.g_light:setFilter("linear", "linear") -- for AA
  scene.g_render = love.graphics.newCanvas(w, h, {format = "rgba32f"})
  scene.g_render:setFilter("linear", "linear") -- for downsampling
  scene.g_luma = love.graphics.newCanvas(w, h, {format = "r8"})

  --- generate bloom buffers (maximum possible iterations)
  scene.bloom_buffers = {}
  do
    local l2w, l2h = math.log(w)/math.log(2), math.log(h)/math.log(2)
    local max_iterations = math.max(math.floor(l2w)-1, math.floor(l2h)-1)

    local bw, bh = w, h
    for i=1, max_iterations do
      bw, bh = math.floor(bw/2), math.floor(bh/2)
      local buffer = love.graphics.newCanvas(bw, bh, {format = "rgba32f"})
      buffer:setFilter("linear", "linear")
      table.insert(scene.bloom_buffers, buffer)
    end
  end

  -- init shaders
  scene.material_shader = love.graphics.newShader(MATERIAL_SHADER)
  scene.translucent_shader = love.graphics.newShader(TRANSLUCENT_SHADER)
  scene.light_shader = love.graphics.newShader(LIGHT_SHADER)
  scene.render_shader = love.graphics.newShader(RENDER_SHADER)
  scene.bloom_extract_shader = love.graphics.newShader(BLOOM_EXTRACT_SHADER)
  scene.bloom_downsample_shader = love.graphics.newShader(BLOOM_DOWNSAMPLE_SHADER)
  scene.bloom_upsample_shader = love.graphics.newShader(BLOOM_UPSAMPLE_SHADER)
  scene.fxaa_shader = love.graphics.newShader(FXAA_SHADER)

  scene:setMaterialColorProfiles("sRGB", "linear")
  scene.render_shader:send("gamma", 2.2)
  scene.render_shader:send("exposure", 1)
  scene.render_shader:send("TMO", 0)
  scene.material_shader:send("scene_depth", scene.depth)
  scene.translucent_shader:send("scene_depth", scene.depth)
  scene.light_shader:send("projection_matrix", scene.projection_matrix)
  scene.light_shader:send("scene_depth", scene.depth)
  scene.light_shader:send("g_depth", scene.g_depth)
  scene.light_shader:send("g_normal", scene.g_normal)
  scene.light_shader:send("g_MR", scene.g_MR)
  scene.light_shader:send("g_emission", scene.g_emission)
  scene.bloom_extract_shader:send("texel_size", {1/scene.w, 1/scene.h})
  scene.fxaa_shader:send("texel_size", {1/scene.w, 1/scene.h})
  scene.fxaa_shader:send("g_luma", scene.g_luma)

  scene:setBloom(0.8, 0.5, 6.5, 0.1)

  scene:setFXAA(0.0312, 0.125, 0.75)
  scene.AA_mode = "none"

  return scene
end

-- Scene

-- Set projection.
-- depth_projection: max depth of the scene or custom projection matrix
--- max depth: orthographic mode, maximum depth distance of the scene (> 0)
--- projection matrix: custom mode, LÖVE table format (row-major)
-- sw, sh: (optional) scene dimensions for orthographic mode (default: w, h)
function Scene:setProjection(depth_projection, sw, sh)
  sw, sh = sw or self.w, sh or self.h

  if type(depth_projection) == "number" then
    self.depth = depth_projection
    self.projection_matrix = {
      2/sw, 0, 0, -1,
      0, -2/sh, 0, 1,
      0, 0, -2/self.depth, -1,
      0, 0, 0, 1
    }
  else
    self.projection_matrix = depth_projection
    self.depth = 0
  end
end

-- Set gamma used for correction.
-- (ignored by "filmic" TMO)
function Scene:setGamma(gamma)
  self.render_shader:send("gamma", gamma)
end

-- Set exposure adjustment.
function Scene:setExposure(exposure)
  self.render_shader:send("exposure", exposure)
end

-- Set tone mapping operator.
-- tmo: tone mapping operator (string)
--- "raw" (scene default)
--- "reinhard"
--- "filmic" (Jim Hejl, Richard Burgess-Dawson)
function Scene:setToneMapping(tmo)
  self.render_shader:send("TMO", TMO[tmo] or 0)
end

-- Configure bloom.
-- Scene default is (0.8,0.5,6.5,0.1).
--
-- threshold: level of brightness
-- knee: 0-1 (0: hard threshold, 1: soft threshold)
-- radius: bloom radius (resolution-independent)
-- intensity: bloom intensity (0 to disable bloom)
-- safe_clamp: (optional) safe color extraction (default: 1e20)
function Scene:setBloom(threshold, knee, radius, intensity, safe_clamp)
  self.bloom_intensity = intensity

  self.bloom_extract_shader:send("bloom_curve", {
    threshold,
    threshold-knee,
    knee*2,
    0.25/(knee+1e-5)
  })
  self.bloom_extract_shader:send("bloom_safe_clamp", safe_clamp or 1e20)

  -- limit iterations by radius
  local l = math.log(self.h)/math.log(2)+radius-8
  self.bloom_iterations = math.max(math.min(#self.bloom_buffers, math.floor(l)), 1) -- clamp l

  -- set upsample scale
  self.bloom_upsample_shader:send("sample_scale", 0.5+l-math.floor(l))
end

-- Set material/translucent textures color profiles.
-- Scene default is "sRGB" for albedo and "linear" for MR.
-- Normal, depth and emission maps must be linear (color wise).
--
-- albedo, MR: color space string ("sRGB" or "linear")
function Scene:setMaterialColorProfiles(albedo, MR)
  self.material_shader:send("m_color_profiles", albedo == "sRGB" and 1 or 0, MR == "sRGB" and 1 or 0)
  self.translucent_shader:send("m_color_profiles", albedo == "sRGB" and 1 or 0, MR == "sRGB" and 1 or 0)
end

-- Set FXAA parameters.
-- contrast_threshold: (scene default: 0.0312)
--- Trims the algorithm from processing darks.
---   0.0833 - upper limit (default, the start of visible unfiltered edges)
---   0.0625 - high quality (faster)
---   0.0312 - visible limit (slower)
--
-- relative_threshold: (scene default: 0.125)
--- The minimum amount of local contrast required to apply algorithm.
---   0.333 - too little (faster)
---   0.250 - low quality
---   0.166 - default
---   0.125 - high quality
---   0.063 - overkill (slower)
--
-- subpixel_blending: (scene default: 0.75)
--- Choose the amount of sub-pixel aliasing removal.
--- This can effect sharpness.
---   1.00 - upper limit (softer)
---   0.75 - default amount of filtering
---   0.50 - lower limit (sharper, less sub-pixel aliasing removal)
---   0.25 - almost off
---   0.00 - completely off
function Scene:setFXAA(contrast_threshold, relative_threshold, subpixel_blending)
  self.fxaa_shader:send("contrast_threshold", contrast_threshold)
  self.fxaa_shader:send("relative_threshold", relative_threshold)
  self.fxaa_shader:send("subpixel_blending", subpixel_blending)
end

-- Set anti-aliasing mode.
-- mode: string
--- "none": disabled (scene default)
--- "FXAA": FXAA 3.11
function Scene:setAntiAliasing(mode)
  self.AA_mode = mode
end

-- Bind canvases and shader.
--
-- The material pass is the process of writing the albedo/shape (RGBA), normal,
-- metalness/roughness and depth of each object of the scene to the G-buffer.
--
-- The albedo texture is to be used with LÖVE draw calls, it defines the albedo
-- and shape (alpha) of the material/object (affected by LÖVE color).
--
-- If the API is too limited, it is better to write a custom material shader and
-- directly call the LÖVE API (ex: ray-marching SDF, different kind of
-- textures, etc.).
function Scene:bindMaterialPass()
  love.graphics.setCanvas({
    self.g_albedo,
    self.g_normal,
    self.g_MR,
    self.g_emission,
    depthstencil = self.g_depth
  })
  love.graphics.clear()

  love.graphics.setShader(self.material_shader)
  love.graphics.setDepthMode("lequal", true)
  love.graphics.setBlendMode("alpha")
end

-- Bind normal map.
-- The normal map must be in view space (X left->right, Y bottom->top, Z far->view).
--
-- normal_map: 3-components texture (RGBA8 format recommended)
function Scene:bindMaterialN(normal_map)
  self.material_shader:send("m_normal", normal_map)
end

-- Bind metalness/roughness map.
-- MR_map: 2-components texture (metalness + roughness, RG8 format recommended)
-- metalness: (optional) metalness factor (default: 1)
-- roughness: (optional) roughness factor (default: 1)
function Scene:bindMaterialMR(MR_map, metalness, roughness)
  local s = self.material_shader
  s:send("m_MR", MR_map)
  s:send("m_MR_args", metalness or 1, roughness or 1)
end

-- Bind depth/emission map.
--
-- In orthographic mode, the depth map is the perpendicular distance to the
-- view plane (not near plane) for each pixel, it can be absolute or normalized (0-1).
-- If normalized, it must be done with this formula: log2(z+1)/log2(max+1).
-- In custom mode, the depth map is written as-is, z and depth_max are not
-- used.
--
-- DE_map: 2-component texture (depth + emission, RG32F format recommended, absolute or normalized)
-- z: (optional) depth of the object (should be positive, default: 0)
-- depth_max: (optional) max distance in the logarithmically normalized depth map (default: 0)
-- emission_factor: (optional) factor for the emission intensity (default: 1)
-- emission_max: (optional) max value in the logarithmically normalized emission map (default: 0)
function Scene:bindMaterialDE(DE_map, z, depth_max, emission_factor, emission_max)
  local s = self.material_shader
  s:send("m_DE", DE_map)
  s:send("m_depth_args", z or 0, depth_max or 0)
  s:send("m_emission_args", emission_factor or 1, emission_max or 0)
end

-- Bind canvases and shader.
-- The light pass is the process of lighting the materials.
function Scene:bindLightPass()
  love.graphics.setDepthMode("always", false)

  -- alpha black pass
  love.graphics.setShader()
  love.graphics.setCanvas(self.g_light)
  love.graphics.clear()
  love.graphics.setColor(0,0,0)
  love.graphics.draw(self.g_albedo)
  love.graphics.setColor(1,1,1)

  -- begin light pass
  love.graphics.setShader(self.light_shader)
  love.graphics.setBlendMode("add")
end

-- (uses LÖVE color)
function Scene:drawAmbientLight(intensity)
  self.light_shader:send("l_type", LIGHTS.AMBIENT)
  self.light_shader:send("l_intensity", intensity)
  love.graphics.draw(self.g_albedo)
end

-- (uses LÖVE color)
function Scene:drawPointLight(x, y, z, radius, intensity)
  self.light_shader:send("l_type", LIGHTS.POINT)
  self.light_shader:send("l_intensity", intensity)
  self.light_shader:send("l_pos_rad", {x,y,z,radius})
  love.graphics.draw(self.g_albedo)
end

-- (uses LÖVE color)
function Scene:drawDirectionalLight(dx, dy, dz, intensity)
  self.light_shader:send("l_type", LIGHTS.DIRECTIONAL)
  self.light_shader:send("l_intensity", intensity)
  self.light_shader:send("l_dir", {dx,dy,dz})
  love.graphics.draw(self.g_albedo)
end

-- Draw emission light pass (uses LÖVE color).
-- intensity: (optional) (default: 1)
function Scene:drawEmissionLight(intensity)
  self.light_shader:send("l_type", LIGHTS.EMISSION)
  self.light_shader:send("l_intensity", intensity or 1)
  love.graphics.draw(self.g_albedo)
end

-- Bind raw light.
-- Used to add raw light on the light buffer with draw calls.
function Scene:bindLight(intensity)
  self.light_shader:send("l_typ", LIGHTS.RAW)
  self.light_shader:send("l_intensity", intensity)
end

-- Bind canvases and shader.
-- The translucent pass is like the material pass, but only the albedo and
-- emission will be used and the depth will not be modified (not a "solid"
-- pass, after the light pass).
function Scene:bindTranslucentPass()
  love.graphics.setCanvas({
    self.g_albedo,
    self.g_emission,
    depthstencil = self.g_depth
  })

  love.graphics.setShader(self.translucent_shader)
  love.graphics.setBlendMode("alpha")
  love.graphics.setDepthMode("lequal", true)
end

-- Same as bindMaterialDE.
function Scene:bindTranslucentDE(DE_map, z, depth_max, emission_factor, emission_max)
  local s = self.translucent_shader
  s:send("m_DE", DE_map)
  s:send("m_depth_args", z or 0, depth_max or 0)
  s:send("m_emission_args", emission_factor or 1, emission_max or 0)
end

-- Final rendering.
-- r,g,b,a: (optional) background color for the render (default: transparent)
function Scene:render(r, g, b, a)
  if not r then r,g,b,a = 0,0,0,0 end

  love.graphics.setBlendMode("alpha")
  love.graphics.setDepthMode("always", false)

  -- draw light buffer to render buffer with background color
  love.graphics.setCanvas(self.g_render)
  love.graphics.setShader()
  love.graphics.clear(r,g,b,a)
  love.graphics.draw(self.g_light)

  if self.bloom_intensity > 0 then
    -- bloom effect
    -- Bright areas are extracted from the render. The result is downsampled with
    -- a 2x2, effectively 4x4 with bilinear filtering, blur filter multiple
    -- times. Then the last frame is upsampled with a 2x2 (4x4) blur filter
    -- multiple times, accumulated to the previous downsampled buffers. The
    -- result is multiplied (intensity) and added to the render.

    --- bloom extract pass
    love.graphics.setCanvas(self.g_light) -- re-use g_light
    love.graphics.clear()
    love.graphics.setShader(self.bloom_extract_shader)
    love.graphics.draw(self.g_render)

    --- downsample steps
    do
      love.graphics.setShader(self.bloom_downsample_shader)
      local previous = self.g_light
      for i=1,self.bloom_iterations do
        local buffer = self.bloom_buffers[i]
        love.graphics.setCanvas(buffer)
        love.graphics.clear()
        self.bloom_downsample_shader:send("texel_size", {1/previous:getWidth(), 1/previous:getHeight()})
        love.graphics.draw(previous, 0, 0, 0, buffer:getWidth()/previous:getWidth(), buffer:getHeight()/previous:getHeight())
        previous = buffer
      end
    end

    --- upsample steps
    do
      love.graphics.setShader(self.bloom_upsample_shader)
      self.bloom_upsample_shader:send("intensity", 1)
      love.graphics.setBlendMode("add")
      local previous = self.bloom_buffers[#self.bloom_buffers]
      for i=self.bloom_iterations-1, 1, -1 do
        local buffer = self.bloom_buffers[i]
        love.graphics.setCanvas(buffer)
        self.bloom_upsample_shader:send("texel_size", {1/previous:getWidth(), 1/previous:getHeight()})
        love.graphics.draw(previous, 0, 0, 0, buffer:getWidth()/previous:getWidth(), buffer:getHeight()/previous:getHeight())
        previous = buffer
      end
    end

    --- final step: upsample + add to render
    do
      local previous, buffer = self.bloom_buffers[1], self.g_render
      self.bloom_upsample_shader:send("intensity", self.bloom_intensity)
      love.graphics.setCanvas(buffer)
      self.bloom_downsample_shader:send("texel_size", {1/previous:getWidth(), 1/previous:getHeight()})
      love.graphics.draw(previous, 0, 0, 0, buffer:getWidth()/previous:getWidth(), buffer:getHeight()/previous:getHeight())
    end
  end

  love.graphics.setBlendMode("alpha")

  -- Anti-aliasing pass
  local target = self.bloom_intensity > 0 and self.g_light or self.g_render
  local source = self.bloom_intensity > 0 and self.g_render or self.g_light

  if self.AA_mode == "FXAA" then
    love.graphics.setCanvas(target, self.g_luma)
    love.graphics.clear()
    love.graphics.setShader(self.render_shader)
    love.graphics.draw(source)

    love.graphics.setCanvas()
    love.graphics.setShader(self.fxaa_shader)
    love.graphics.draw(target)
    love.graphics.setShader()
  else -- no AA
    love.graphics.setCanvas()
    love.graphics.setShader(self.render_shader)
    love.graphics.draw(source)
    love.graphics.setShader()
  end
end

return DPBR
