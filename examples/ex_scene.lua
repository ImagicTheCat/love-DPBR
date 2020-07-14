local DPBR = require("love-DPBR")

local scene = DPBR.newScene(1280,720)
scene:setProjection2D(50, "log", 10, 10*9/16)
scene:setToneMapping("filmic")
scene:setAntiAliasing("FXAA")
scene:setAmbientBRDF(love.graphics.newImage("BRDF_LUT.exr"))

local t_albedo = love.graphics.newImage("scene_albedo.jpg")
local t_normal = love.graphics.newImage("scene_normal.png")
local t_DE = love.graphics.newImage("scene_DE.exr")
local t_MRA = love.graphics.newImage("scene_MRA.jpg")
local env_diffuse = love.graphics.newCubeImage("env_diffuse.dds")
local env_specular = love.graphics.newCubeImage("env_specular.dds", {mipmaps = true})
env_specular:setMipmapFilter("linear")

local function tick(dt)
end

local function draw()
  local time = love.timer.getTime()
  local x,y = love.mouse.getPosition()

  scene:bindMaterialPass()
  scene:bindMaterialN(t_normal)
  scene:bindMaterialMRA(t_MRA)
  scene:bindMaterialDE(t_DE)
  love.graphics.draw(t_albedo,0,0)

  scene:bindLightPass()
  scene:drawEnvironmentLight(env_diffuse, env_specular, 1)
  scene:drawEmissionLight()

  love.graphics.setColor(HSL(time/50%1, 1, 0.5))
  scene:drawPointLight(x/scene.w*10,y/scene.h*10*9/16,2,50,25)
  scene:drawPointLight(x/scene.w*10,y/scene.h*10*9/16,15,50,25)
  love.graphics.setColor(1,1,1)

  scene:bindBackgroundPass()
  scene:bindBlendPass()
  scene:render()
end

return tick, draw, "3D baked scene.\n\nModel by Andrew Maximov.\n(http://artisaverb.info/PBT.html)"
