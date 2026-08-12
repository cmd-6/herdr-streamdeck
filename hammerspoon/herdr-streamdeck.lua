local module = {}

local home = os.getenv("HOME")
local statePath = home .. "/Library/Caches/herdr-streamdeck/state.json"
local tokenPath = home .. "/.config/herdr-streamdeck/virtual-token"
local api = "http://127.0.0.1:17373"
local keySize = 84
local gap = 10
local padding = 14
local rows = 3
local columns = 5
local canvas = nil
local visible = false
local pinned = false
local pressedAt = {}
local keyRevisions = {}
local escapeHotkey = nil

local function readFile(path)
	local handle = io.open(path, "r")
	if not handle then
		return nil
	end
	local value = handle:read("*a")
	handle:close()
	return value
end

local function post(index, state)
	local token = readFile(tokenPath)
	if not token then
		return
	end
	token = token:gsub("%s+$", "")
	hs.http.asyncPost(
		api .. "/keys/" .. index .. "/" .. state,
		"",
		{ ["Authorization"] = "Bearer " .. token },
		function() end
	)
end

local function frameFor(index)
	local row = math.floor(index / columns)
	local column = index % columns
	return {
		x = padding + column * (keySize + gap),
		y = padding + row * (keySize + gap),
		w = keySize,
		h = keySize,
	}
end

local function hide()
	if not visible or not canvas then
		return
	end
	visible = false
	module.poller:stop()
	if escapeHotkey then
		escapeHotkey:disable()
	end
	canvas:hide(0.12)
end

local function refresh()
	if not visible or not canvas then
		return
	end
	local encoded = readFile(statePath)
	if not encoded then
		return
	end
	local ok, state = pcall(hs.json.decode, encoded)
	if not ok or not state or not state.keys then
		return
	end
	for index = 0, (#state.keys - 1) do
		local revision = state.key_revisions[index + 1]
		if revision ~= keyRevisions[index] then
			local image = hs.image.imageFromPath(state.keys[index + 1])
			if image then
				canvas[index + 2].image = image
				keyRevisions[index] = revision
			end
		end
	end
end

local function buildCanvas()
	local width = 2 * padding + columns * keySize + (columns - 1) * gap
	local height = 2 * padding + rows * keySize + (rows - 1) * gap
	local screen = hs.screen.mainScreen():frame()
	local frame = {
		x = screen.x + (screen.w - width) / 2,
		y = screen.y + 18,
		w = width,
		h = height,
	}
	canvas = hs.canvas.new(frame)
	canvas:appendElements({
		type = "rectangle",
		action = "fill",
		fillColor = { red = 0.035, green = 0.035, blue = 0.045, alpha = 0.96 },
		roundedRectRadii = { xRadius = 18, yRadius = 18 },
	})
	for index = 0, rows * columns - 1 do
		canvas:appendElements({
			id = "key-" .. index,
			type = "image",
			frame = frameFor(index),
			imageScaling = "scaleToFit",
			trackMouseDown = true,
			trackMouseUp = true,
		})
	end
	canvas:mouseCallback(function(_, message, identifier)
		local index = tonumber(identifier:match("key%-(%d+)$"))
		if not index then
			return
		end
		if message == "mouseDown" then
			pressedAt[index] = hs.timer.secondsSinceEpoch()
			post(index, "down")
		elseif message == "mouseUp" then
			post(index, "up")
			local duration = hs.timer.secondsSinceEpoch() - (pressedAt[index] or 0)
			pressedAt[index] = nil
			if duration >= 0.45 then
				pinned = true
			elseif not pinned then
				hs.timer.doAfter(0.12, hide)
			end
		end
	end)
	canvas:clickActivating(false)
	canvas:behaviorAsLabels({ "canJoinAllSpaces", "fullScreenAuxiliary", "stationary" })
	canvas:level(hs.canvas.windowLevels.popUpMenu)
end

local function show()
	if not readFile(statePath) then
		hs.alert.show("Herdr virtual deck is not running")
		return
	end
	if canvas then
		canvas:delete()
	end
	keyRevisions = {}
	pinned = false
	buildCanvas()
	visible = true
	refresh()
	canvas:show(0.12)
	canvas:bringToFront(false)
	escapeHotkey:enable()
	module.poller:start()
end

function module.toggle()
	if visible then
		hide()
	else
		show()
	end
end

module.poller = hs.timer.new(0.1, refresh)
module.hotkey = hs.hotkey.bind({ "alt" }, "space", module.toggle)
escapeHotkey = hs.hotkey.new({}, "escape", function()
	if visible then
		hide()
	end
end)
module.escape = escapeHotkey

return module
