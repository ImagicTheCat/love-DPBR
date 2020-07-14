-- https://github.com/ImagicTheCat/love-DPBR
-- MIT license (see LICENSE or src/love-DPBR.lua)

-- Bake some useful textures.
-- requirements: LuaJIT, MGL, lua-vips
-- params:
--- BRDF <output .tiff 32-bit file> [size] [samples]
local bit = require("bit")
local ffi = require("ffi")
local mgl = require("MGL")
local vips = require("vips")
mgl.gen_vec(2); mgl.gen_vec(3); mgl.gen_vec(4)
mgl.gen_mat(3); mgl.gen_mat(4)
local bor, band, bswap, lshift, rshift = bit.bor, bit.band, bit.bswap, bit.lshift, bit.rshift
local cross = mgl.getOp("cross", "vec3", "vec3")
local normalize_vec3 = mgl.getOp("normalize", "vec3")
local dot_vec3 = mgl.getOp("dot", "vec3", "vec3")
local vec3s = mgl.getOp("vec3", "number", "number", "number")
local vec2s = mgl.getOp("vec2", "number", "number")
local mul_mat3v = mgl.getOp("mul", "mat3", "vec3")
local mat3v = mgl.getOp("mat3", "vec3", "vec3", "vec3")

-- return mat3 transform from normal
local ax, ay = mgl.vec3(1,0,0), mgl.vec3(0,1,0)
local function normalBase(n)
  local up = math.abs(n.y) < 0.999 and ay or ax
  local t = normalize_vec3(cross(up, n))
  local b = cross(n, t)
  return mat3v(t,b,n)
end

-- Reverse bits order.
-- (swap blocks two by two recursively)
-- n: 32-bits unsigned integer value
-- return 32-bits unsigned integer value
local function reverse32b(n)
  n = bswap(n)
  n = bor(lshift(band(n, 0x0f0f0f0f), 4), rshift(band(n, 0xf0f0f0f0), 4))
  n = bor(lshift(band(n, 0x33333333), 2), rshift(band(n, 0xcccccccc), 2))
  n = bor(lshift(band(n, 0x55555555), 1), rshift(band(n, 0xaaaaaaaa), 1))
  return n < 0 and n+0x100000000 or n
end

-- van der Corput base 2 sequence.
-- i: element index
-- return sequence element (0-1)
local function vdcorput2(i) return reverse32b(i)*0x1p-32 end

-- Hammersley sequence.
-- i: element index
-- n: total number of elements
-- return (x,y) 2D sequence element (0-1)
local function hammersley(i, n) return i/n, vdcorput2(i) end

-- Generate sample vector from Trowbridge-Reitz GGX normal distribution
-- function and roughness.
--
-- p: sequence point (vec2)
-- n: normal (vec3)
-- a: effective roughness
-- return vec3
local function importanceSampleGGX(p, n, a)
  -- spherical coordinates
  local phi = 2*math.pi*p[1]
  local cos_theta = math.sqrt((1-p[2])/(1+(a*a-1)*p[2]))
  local sin_theta = math.sqrt(1-cos_theta*cos_theta)
  -- cartesian coordinates
  local v = vec3s(math.cos(phi)*sin_theta, math.sin(phi)*sin_theta, cos_theta)
  -- world space
  return mul_mat3v(normalBase(n))*v
end

local function geometrySchlickGGX(NdotV, k)
  return NdotV/(NdotV*(1-k)+k)
end

local function geometrySmith(NdotV, NdotL, k)
  return geometrySchlickGGX(NdotV, k)*geometrySchlickGGX(NdotL, k)
end

-- Compute BRDF F0 scale and bias integrals.
-- a: squared roughness
-- k: geometry parameter (roughness*roughness/2 for ambient/IBL)
-- samples: number of samples for quasi-Monte Carlo integration
-- return (scale, bias)
local function integrateBRDF(NdotV, a, k, samples)
  local v = vec3s(math.sqrt(1-NdotV*NdotV), 0, NdotV)
  local n = vec3s(0,0,1)
  local A,B = 0,0
  for i=1,samples do
    local p = vec2s(hammersley(i, samples))
    local h = importanceSampleGGX(p, n, a)
    local l = normalize_vec3(2*dot_vec3(v,h)*h-v)
    local NdotL, NdotH, VdotH = math.max(l.z, 0), math.max(h.z, 0), math.max(dot_vec3(v,h), 0)
    if NdotL > 0 then
      local G = geometrySmith(NdotV, NdotL, k)
      local G_vis = (G*VdotH)/(NdotH*NdotV)
      local Fc = (1-VdotH)^5
      A = A+(1-Fc)*G_vis
      B = B+Fc*G_vis
    end
  end
  return A/samples, B/samples
end

-- Compute diffuse irradiance integral.
-- n: normal/lookup vector (vec3)
-- delta: sampling delta
-- lookup(v): lookup callback
--- v: lookup vector for a sample (vec3)
--- return sample radiance (vec3)
local function integrateDiffuse(n, delta, lookup)
  local tr = normalBase(n) -- tangent to world space
  local irr = vec3s(0,0,0)
  local phi = 0
  local samples = 0
  while phi < math.pi*2 do
    local theta = 0
    while theta < math.pi/2 do
      local v = vec3s(math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi), math.cos(theta))
      irr = irr+lookup(mul_mat3v(tr,v))*math.cos(theta)*math.sin(theta)
      samples = samples+1
      theta = theta+delta
    end
    phi = phi+delta
  end
  return irr*math.pi/samples
end

-- cubemap constants
local cubemap_faces = {} -- list of {name, mat3 transform}
do
  local ay = mgl.vec3(0,1,0)
  local ax = mgl.vec3(1,0,0)
  table.insert(cubemap_faces, {"pZ", mgl.mat3(1)})
  table.insert(cubemap_faces, {"pX", mgl.mat3(mgl.rotate(ay, math.pi/2))})
  table.insert(cubemap_faces, {"mZ", mgl.mat3(mgl.rotate(ay, math.pi))})
  table.insert(cubemap_faces, {"mX", mgl.mat3(mgl.rotate(ay, -math.pi/2))})
  table.insert(cubemap_faces, {"mY", mgl.mat3(mgl.rotate(ax, math.pi/2))})
  table.insert(cubemap_faces, {"pY", mgl.mat3(mgl.rotate(ax, -math.pi/2))})
end

-- Convert lookup vector to equirectangular image coordinates.
-- v: lookup vector
-- return (x,y) (0-1, top-left origin)
local function equirectangularCoords(v)
  return (math.pi+math.atan2(v[3], v[1]))/(math.pi*2), math.acos(v[2])/math.pi
end

-- run

local args = {...}
local mode = args[1]
if mode == "BRDF" then
  local out = args[2] or "BRDF_LUT.tiff"
  local size, samples = tonumber(args[3]) or 512, tonumber(args[4]) or 1024
  print("output: "..out..", size: "..size..", samples: "..samples)
  local data = ffi.new("float[?]", size*size*2)
  for i=0,size-1 do
    for j=0,size-1 do
      local NdotV, roughness = (i+0.5)/size, 1-(j+0.5)/size
      local a = roughness*roughness
      local k = a/2
      data[(j*size+i)*2], data[(j*size+i)*2+1] = integrateBRDF(NdotV, a, k, samples)
    end
  end
  local img = vips.Image.new_from_memory(data, size, size, 2, "float")
  img:bandjoin({0,1}):write_to_file(out)
--[[
-- experimental implementation
elseif mode == "diffuse" then
  -- diffuse <input equirectangular format> <output format> [size] [samples factor]
  --- equirectangular format: floating-point format or .hdr
  --- output format: "tiff" or "hdr"

  -- args
  if #args < 3 then error("invalid arguments") end
  local input, format, size, samples_factor = args[2], args[3], tonumber(args[4]) or 512, tonumber(args[5]) or 1
  local delta = 0.025/samples_factor
  local f_base, f_ext = string.match(input, "^(.+)%.(.-)$")
  -- load image
  local in_img = vips.Image.new_from_file(input)
  if in_img:get("coding") == "rad" then in_img = in_img:rad2float() end
  local w,h = in_img:width(), in_img:height()
  local bands = in_img:bands()
  if w ~= h*2 or bands < 3 or in_img:format() ~= "float" then error("invalid image format") end
  local in_data_ref = in_img:write_to_memory()
  local in_data = ffi.cast("float*", in_data_ref)
  -- lookup function
  local function lookup(v) -- equirectangular lookup
    local x, y = equirectangularCoords(v)
    x, y = math.floor(x*(w-1)), math.floor(y*(h-1))
    return vec3s(in_data[(y*w+x)*bands], in_data[(y*w+x)*bands+1], in_data[(y*w+x)*bands+2])
  end

  local out_data = ffi.new("float[?]", size*size*3)
  for _, face in ipairs(cubemap_faces) do
    local face_name, tr = face[1], face[2]
    local out = f_base.."_"..face_name.."."..format
    print("output: "..out..", size: "..size)
    for i=0,size-1 do
      for j=0,size-1 do
        -- compute cubemap vector (based on size 2 cube, center as origin)
        local n = mul_mat3v(tr, normalize_vec3(vec3s(i/size*2-1, (1-j/size)*2-1, 1)))
        -- integrate
        local irr = integrateDiffuse(n, delta, lookup)
        out_data[(j*size+i)*3], out_data[(j*size+i)*3+1], out_data[(j*size+i)*3+2] = irr[1], irr[2], irr[3]
      end
    end
    local img = vips.Image.new_from_memory(out_data, size, size, 3, "float")
    if format == "hdr" then img:float2rad():write_to_file(out)
    else img:bandjoin({1}):write_to_file(out) end
  end
elseif mode == "specular" then
--]]
else error("invalid mode") end
