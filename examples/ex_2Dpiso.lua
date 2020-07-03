local DPBR = require("love-DPBR")

local base = 128 -- x diagonal of base tile in pixels
local unit = base/math.sqrt(2) -- pixels per meter
local sw, sh = 1280/unit, 720/unit
local scene = DPBR.newScene(1280, 720)
scene:setProjection2D(100, "log", sw, sh)
scene:setToneMapping("filmic")
scene:setAntiAliasing("FXAA")
scene:setBloom(0.8,0.5,6.5,0.05)

local t_MR = love.graphics.newImage("default_MR.png")

-- Convert pseudo-iso 3D coordinates to screen coordinates (top->left, bottom->top, view->far).
-- (0,0,0) => (0,0,0)
--
-- depth xy = sin θ * cos φ
-- depth z = -cos θ
-- θ = PI/3, φ = PI/4
--
-- return x, y, z
--- z: depth in meters from view plane
local function worldToScreen(x,y,z)
  return (y-x)*base/2, (x+y)*base/4+z*base/2, (x+y)*math.sin(math.pi/3)*math.cos(math.pi/4)-z*0.5
end

-- Inverse of worldToScreen.
-- return x, y, z
local function screenToWorld(x,y,z)
  local f = math.sin(math.pi/3)*math.cos(math.pi/4)
  local wz = (2*z-8*y*f/base)/(-1-4*f)
  local wy = (2*y+x)/base-wz
  return -2*x/base+wy, wy, wz
end

-- Same as screenToWorld, but wz is the world z coordinate.
-- return x, y, z
local function pointerToWorld(x,y,wz)
  local f = math.sin(math.pi/3)*math.cos(math.pi/4)
  local wy = (2*y+x)/base-wz
  return -2*x/base+wy, wy, wz
end

local function loadTile(name)
  local t = {}
  t.albedo = love.graphics.newImage("piso_"..name.."_albedo.png")
  t.normal = love.graphics.newImage("piso_"..name.."_normal.png")
  t.DE = love.graphics.newImage("piso_"..name.."_DE.exr")
  return t
end

local quads = {}
for i=0,3 do
  table.insert(quads, love.graphics.newQuad(base*i, 0, base, base, base*4, base))
end

local ground_tile = loadTile("ground")
local tiles = {}
table.insert(tiles, loadTile("cube"))
table.insert(tiles, loadTile("cylinder"))
table.insert(tiles, loadTile("sphere"))
table.insert(tiles, loadTile("suzanne"))

local cursor_tile = tiles[3]

local map = {} -- list of (tile,x,y,depth)
local map_size = 10
local depth_bias = 1

-- generation
for i=0,map_size-1 do
  for j=0,map_size-1 do
    local x,y,z = worldToScreen(i,j,0)
    local dir = math.random(1,4)

    table.insert(map, {ground_tile, dir, x, y, z, 1, 1, 1, 0, 0.5})

    -- level 1
    local tile = tiles[math.random(1, #tiles+3)] -- distribution bias
    if tile then
      local x,y,z = worldToScreen(i,j,1)
      local r,g,b = math.random(), math.random(), math.random()
      local metalness, roughness = math.random(0,1), math.random()
      table.insert(map, {tile, dir, x, y, z, r, g, b, metalness, roughness})
    end
  end
end

local function tick(dt)
end

local function draw()
  local mx,my = love.mouse.getPosition()
  local time = love.timer.getTime()

  scene:bindMaterialPass()

  -- draw map
  for _, mtile in ipairs(map) do
    local tile, dir, x, y, z, r, g, b, metalness, roughness = unpack(mtile)
    love.graphics.setColor(r,g,b)
    scene:bindMaterialN(tile.normal)
    scene:bindMaterialMR(t_MR, metalness, roughness)
    scene:bindMaterialDE(tile.DE, z+depth_bias, 0)
    love.graphics.draw(tile.albedo, quads[dir], scene.w/2-base/2+x, scene.h-y-tile.albedo:getHeight())
  end

  -- draw cursor
  local wx,wy,wz = pointerToWorld(mx-scene.w/2, scene.h-my, 1)
  local cx,cy,cz = worldToScreen(wx,wy,wz)
  love.graphics.setColor(HSL(time/50%1, 1, 0.65))

  scene:bindMaterialN(cursor_tile.normal)
  scene:bindMaterialMR(t_MR, 0, 0)
  scene:bindMaterialDE(cursor_tile.DE, cz+depth_bias, 15)
  love.graphics.draw(cursor_tile.albedo, quads[1], scene.w/2-base/2+cx, scene.h-cy-cursor_tile.albedo:getHeight())
  love.graphics.setColor(1,1,1)

  scene:bindLightPass()
  scene:drawEmissionLight()
  love.graphics.setColor(HSL(time/50%1, 1, 0.65))
  scene:drawPointLight((scene.w/2+cx)/unit,(scene.h-cy-base/2)/unit,cz,100,20)
  love.graphics.setColor(1,1,1)

  scene:render(0,0,0,1)
end

return tick, draw, "2D pseudo-isometric (dimetric)."
