function HSL(h,s,l,a)
	if s<=0 then return l,l,l,a end
	h, s, l = h*6, s, l
	local c = (1-math.abs(2*l-1))*s
	local x = (1-math.abs(h%2-1))*c
	local m,r,g,b = (l-.5*c), 0,0,0
	if h < 1     then r,g,b = c,x,0
	elseif h < 2 then r,g,b = x,c,0
	elseif h < 3 then r,g,b = 0,c,x
	elseif h < 4 then r,g,b = 0,x,c
	elseif h < 5 then r,g,b = x,0,c
	else              r,g,b = c,0,x
	end return r+m,g+m,b+m,a
end

function createPixelTexture(format, ...)
  local data = love.image.newImageData(1, 1, format)
  data:setPixel(0, 0, ...)
  return love.graphics.newImage(data)
end

local examples = {
  "ex_spheres",
  "ex_scene",
  "ex_2Dside",
  "ex_2Dpiso",
  "ex_SDF_spheres",
  "ex_SDF_mandelbulb"
}

local index = 1
local text_info
local text_stats
local app

function love.load()
  app = love.filesystem.load(examples[index]..".lua")()
  text_info = love.graphics.newText(love.graphics.getFont(), "[SPACE] to switch examples\n"..app.info)
  text_stats = love.graphics.newText(love.graphics.getFont())
end

function love.update(dt)
  app.tick(dt)
  text_stats:set(love.timer.getFPS().." FPS\n"..(app.stats or ""))
end

function love.draw()
  app.draw()
  -- info
  love.graphics.setColor(0,0,0,0.75)
  love.graphics.rectangle("fill", 0,0, text_info:getWidth()+8, text_info:getHeight()+8)
  love.graphics.setColor(1,1,1)
  love.graphics.draw(text_info, 4, 4)

  -- stats
  love.graphics.setColor(0,0,0,0.75)
  love.graphics.rectangle("fill", 1280-text_stats:getWidth()-8, 0, text_stats:getWidth()+8, text_stats:getHeight()+8)
  love.graphics.setColor(1,1,1)
  love.graphics.draw(text_stats, 1280-text_stats:getWidth()-4, 4)
end

function love.keypressed(keycode, scancode, isrepeat)
  if keycode == "space" then
    index = index%#examples+1
    if app.close then app.close() end
    app = love.filesystem.load(examples[index]..".lua")()
    text_info:set("[SPACE] to switch examples\n"..app.info)
  else
    if app.keypressed then app.keypressed(keycode, scancode, isrepeat) end
  end
end

function love.mousemoved(...)
  if app.mousemoved then app.mousemoved(...) end
end
