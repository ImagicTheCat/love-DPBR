local DPBR = require("love-DPBR")
local mgl = require("MGL")
mgl.gen_vec(3); mgl.gen_mat(3)
mgl.gen_mat(4)

local scene = DPBR.newScene(1280,720)
scene:setProjection2D(50, "log", 10, 10*9/16)
scene:setToneMapping("filmic")
scene:setAntiAliasing("FXAA")
scene:setAmbientBRDF(love.graphics.newImage("BRDF_LUT.exr"))

local t_albedo = love.graphics.newImage("sphere_albedo.png")
local t_normal = love.graphics.newImage("sphere_normal.png")
local t_MRA = createPixelTexture("rgba8", 1, 1, 1)
local t_DE = love.graphics.newImage("sphere_DE.exr")
local env_diffuse = love.graphics.newCubeImage("env_diffuse.dds")
local env_specular = love.graphics.newCubeImage("env_specular.dds", {mipmaps = true})
env_specular:setMipmapFilter("linear")

local ax = mgl.vec3(1,0,0)
local ay = mgl.vec3(0,1,0)

local function draw_line(x, y, count, color, metalness)
  scene:bindMaterialN(t_normal)
  scene:bindMaterialDE(t_DE)
  love.graphics.setColor(color)
  for i=0,count-1 do
    scene:bindMaterialMRA(t_MRA, metalness, i/(count-1))
    love.graphics.draw(t_albedo, x+i*120, y)
  end
  love.graphics.setColor(1,1,1)
end

local function tick(dt)
end

local function draw()
  local x,y = love.mouse.getPosition()
  local time = love.timer.getTime()

  scene:bindMaterialPass()
  -- white, black, red dielectric/metallic spheres with variable roughness
  draw_line(40, 0, 10, {1,1,1}, 0)
  draw_line(40, 120, 10, {1,1,1}, 1)
  draw_line(40, 120*2, 10, {0,0,0}, 0)
  draw_line(40, 120*3, 10, {0,0,0}, 1)
  draw_line(40, 120*4, 10, {1,0,0}, 0)
  draw_line(40, 120*5, 10, {1,0,0}, 1)

  scene:bindLightPass()
  local tr = mgl.mat3(mgl.rotate(ax, (y/720-0.5)*math.pi)*mgl.rotate(ay, (x/1280-0.5)*math.pi*2))
  scene:drawEnvironmentLight(env_diffuse, env_specular, 1, tr)

  scene:bindBackgroundPass()
  love.graphics.clear(0,0,0,1)
  scene:bindBlendPass()
  scene:render()
end

return tick, draw, "Spheres."
