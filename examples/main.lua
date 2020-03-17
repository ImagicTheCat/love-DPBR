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

local examples = {
  "ex_scene",
  "ex_2Dside",
  "ex_2Dpiso"
}

local index = 1
local tick, draw
local text

function love.load()
  local desc
  tick, draw, desc = love.filesystem.load(examples[index]..".lua")()
  text = love.graphics.newText(love.graphics.getFont(), "[SPACE] to switch examples\n"..desc)
end

function love.update(dt)
  tick(dt)
end

function love.draw()
  draw()
  love.graphics.setColor(0,0,0,0.75)
  love.graphics.rectangle("fill", 0,0, text:getWidth()+8, text:getHeight()+8)
  love.graphics.setColor(1,1,1)
  love.graphics.draw(text, 4, 4)
end

function love.keypressed(keycode, scancode, isrepeat)
  if keycode == "space" then
    index = index%#examples+1
    local desc
    tick, draw, desc = love.filesystem.load(examples[index]..".lua")()
    text:set("[SPACE] to switch examples\n"..desc)
  end
end
