local DPBR = require("love-DPBR")
local mgl = require("MGL")
mgl.gen_mat(3); mgl.gen_vec(3)
mgl.gen_mat(4); mgl.gen_vec(4)

local app = {info = [[Ray-marching SDF - Spheres.

[WASD] + MOUSE to move
[PAGEUP] +max steps
[PAGEDOWN] -max steps]]}


local proj = mgl.perspective(math.pi/2, 16/9, 0.05, 50)
local inv_proj = mgl.inverse(proj)
local scene = DPBR.newScene(1280,720)
scene:setProjection(proj, inv_proj)
scene:setDepth("raw")
scene:setToneMapping("filmic")
scene:setAntiAliasing("FXAA")
scene:setAmbientBRDF(love.graphics.newImage("BRDF_LUT.exr"))

local shader = love.graphics.newShader([[
#pragma language glsl3

uniform mat4 proj, inv_proj, view, inv_view;
uniform float max_depth;
uniform int max_steps;

float sphereSDF(vec3 p, float r){ return length(-p)-r; }

float torusSDF(vec3 p, vec2 t)
{
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}

vec3 repeatDomain(vec3 p, vec3 c){ return mod(p+c*0.5,c)-c*0.5; }

float sceneSDF(vec3 p)
{
  return min(sphereSDF(repeatDomain(p, vec3(5)), 1),
    torusSDF(repeatDomain(p, vec3(5)), vec2(1.5,0.1)));
}

// return intersection position
vec3 rayMarch(vec3 start, vec3 ray, int max_steps, float epsilon, float max_depth)
{
  float depth = 0;
  vec3 p = start;
  // ray-march scene
  for(int i = 0; i < max_steps; i++){
    float dist = sceneSDF(p);
    if(dist < epsilon)
      return p;
    else{
      depth += dist;
      p += ray*dist;
    }

    if(depth >= max_depth)
      return start+ray*max_depth;
  }
  return start+ray*max_depth;
}

vec3 computeNormal(vec3 p)
{
  const float h = 0.0001; // replace by an appropriate value
  const vec2 k = vec2(1,-1);
  return normalize(k.xyy * sceneSDF(p + k.xyy*h)
    + k.yyx * sceneSDF(p + k.yyx*h)
    + k.yxy * sceneSDF(p + k.yxy*h)
    + k.xxx * sceneSDF(p + k.xxx*h));
}

void effect()
{
  // compute ray
  vec3 ndc = vec3(love_PixelCoord/love_ScreenSize.xy, 0);
  ndc.y = 1.0-ndc.y;
  ndc = ndc*2.0-vec3(1);
  vec4 v = inv_proj*vec4(ndc, 1);
  v /= v.w;
  vec3 w_start = vec3(inv_view*v);
  vec3 ray = normalize(v.xyz);
  vec3 w_ray = mat3(inv_view)*ray;

  vec3 p = rayMarch(w_start, w_ray, max_steps, 0.001, max_depth);
  if(length(p-w_start) >= max_depth-1) discard;

  vec3 n = mat3(view)*computeNormal(p);

  vec4 p_ndc = proj*view*vec4(p, 1);
  p_ndc /= p_ndc.w;

  // depth
  gl_FragDepth = p_ndc.z;
  // albedo
  love_Canvases[0] = vec4(0,0.5,0.25,1);
  // normal
  love_Canvases[1] = vec4((n*vec3(1,-1,-1)+vec3(1))/2, 1);
  // MRA
  love_Canvases[2] = vec4(0,0.25,1,1);
  // emission
  love_Canvases[3] = vec4(0,0,0,1);
}
]])
shader:send("proj", proj)
shader:send("inv_proj", inv_proj)
shader:send("max_depth", 50)

local max_steps = 128
shader:send("max_steps", max_steps)

local ax = mgl.vec3(1,0,0)
local ay = mgl.vec3(0,1,0)
local az = mgl.vec3(0,0,1)
local camera = {p = mgl.vec3(0,0,-2.5), phi = 0, theta = 0}
local function update_cam()
  camera.model = mgl.translate(camera.p)*mgl.rotate(ay, camera.phi)*mgl.rotate(ax, camera.theta)
  camera.view = mgl.inverse(camera.model)
end
update_cam()

local speed = 5
love.mouse.setRelativeMode(true)

function app.tick(dt)
  -- camera translation
  local dir = mgl.mat3(camera.model)*mgl.vec3(0,0,-1)
  local side = mgl.normalize(mgl.cross(dir, ay))

  local is_down = love.keyboard.isScancodeDown
  local vdir = ((is_down("w") and 1 or 0)+(is_down("s") and -1 or 0))*speed*dt
  local vside = ((is_down("a") and -1 or 0)+(is_down("d") and 1 or 0))*speed*dt
  camera.p = camera.p+vdir*dir+vside*side

  -- compute camera transform
  update_cam()
  shader:send("view", camera.view)
  shader:send("inv_view", camera.model)

  app.stats = "max steps: "..max_steps
end

function app.keypressed(keycode, scancode, isrepeat)
  if keycode == "pageup" then
    max_steps = max_steps*2
    shader:send("max_steps", max_steps)
  elseif keycode == "pagedown" then
    max_steps = math.ceil(max_steps/2)
    shader:send("max_steps", max_steps)
  end
end

function app.mousemoved(x, y, dx, dy)
  camera.phi = camera.phi-dx*math.pi*1e-3
  camera.theta = math.min(math.max(camera.theta-dy*math.pi*1e-3, -math.pi/2*0.9), math.pi/2*0.9)
end

function app.draw()
  local time = love.timer.getTime()
  local x,y = love.mouse.getPosition()

  scene:bindMaterialPass()
  love.graphics.setShader(shader)
  love.graphics.rectangle("fill", 0, 0, 1280, 720)
  love.graphics.setShader()

  scene:bindLightPass()
  scene:drawEmissionLight()
  scene:drawPointLight(0,0,0.1,100,25)

  scene:bindBackgroundPass()
  love.graphics.clear(0,0,0,1)
  scene:bindBlendPass()
  scene:render()
end

function app.close()
  love.mouse.setRelativeMode(false)
end

return app
