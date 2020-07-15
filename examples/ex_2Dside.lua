local DPBR = require("love-DPBR")

local app = {info = "2D side.\n\nModel by Andrew Maximov.\n(http://artisaverb.info/PBT.html)"}
local sw, sh = 1280/100, 720/100
local scene = DPBR.newScene(1280, 720)
scene:setProjection2D(10, "log", sw, sh)
scene:setToneMapping("filmic")
scene:setAntiAliasing("FXAA")
scene:setAmbientBRDF(love.graphics.newImage("BRDF_LUT.exr"))

local t_albedo = love.graphics.newImage("object_albedo.png")
local t_normal = love.graphics.newImage("object_normal.png")
local t_DE = love.graphics.newImage("object_DE.exr")
local t_MRA = love.graphics.newImage("object_MRA.png")

love.physics.setMeter(1)
local world = love.physics.newWorld(0,9.8)
local object_shape = love.physics.newPolygonShape(
  0-0.75, 0.21-0.4,
  1.12-0.75, 0.21-0.4,
  1.28-0.75, 0.62-0.4,
  1.14-0.75, 0.62-0.4,
  0.74-0.75, 0.35-0.4,
  0-0.75, 0.31-0.4
)
local w_border = love.physics.newRectangleShape(20,1)
local h_border = love.physics.newRectangleShape(1,20)

local t_border = love.physics.newBody(world, sw/2, -0.5)
love.physics.newFixture(t_border, w_border)
local b_border = love.physics.newBody(world, sw/2, sh+0.5)
love.physics.newFixture(b_border, w_border)
local l_border = love.physics.newBody(world, -0.5, sh/2)
love.physics.newFixture(l_border, h_border)
local r_border = love.physics.newBody(world, sw+0.5, sh/2)
love.physics.newFixture(r_border, h_border)

local s_cursor = love.physics.newCircleShape(0.4)
local cursor = love.physics.newBody(world, 0, 0, "dynamic")
love.physics.newFixture(cursor, s_cursor)
local cursor_joint = love.physics.newMouseJoint(cursor,0,0)

local objects = {}
for i=1,100 do
  local obj = love.physics.newBody(world, math.random(0, sw), math.random(0, sh), "dynamic")
  obj:setMass(5)
  love.physics.newFixture(obj, object_shape)
  table.insert(objects, obj)
end


function app.tick(dt)
  local x,y = love.mouse.getPosition()
  cursor_joint:setTarget(x/100, y/100)
  world:update(dt)
end

function app.draw()
  local x,y = love.mouse.getPosition()

  scene:bindMaterialPass()
  scene:bindMaterialN(t_normal)
  scene:bindMaterialMRA(t_MRA)
  scene:bindMaterialDE(t_DE)

  for _, body in ipairs(objects) do
    love.graphics.push()
    love.graphics.translate(body:getX()*100, body:getY()*100)
    love.graphics.rotate(body:getAngle())
    love.graphics.draw(t_albedo, -t_albedo:getWidth()/2, -t_albedo:getHeight()/2)
    love.graphics.pop()
  end

  scene:bindLightPass()
  scene:drawAmbientLight(0.075)
  scene:drawEmissionLight()

  love.graphics.setColor(HSL(0.04,1,0.52))
  scene:drawPointLight(x/scene.w*sw, y/scene.h*sh,0,50,250)
  love.graphics.setColor(1,1,1)

  scene:bindBackgroundPass()
  love.graphics.clear(0,0,0,1)
  scene:bindBlendPass()
  scene:render()
end

return app
