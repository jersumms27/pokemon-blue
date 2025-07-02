-- A = 0
-- B = 1
-- SELECT = 2
-- START = 3
-- RIGHT = 4
-- LEFT = 5
-- UP = 6
-- DOWN = 7
-- R = 8
-- L = 9

local action_path = 'C:/Users/jerem/Pokemon Blue DQN/communication/action.txt'
local state_path = 'C:/Users/jerem/Pokemon Blue DQN/communication/state.txt'
local reset_path = 'C:/Users/jerem/Pokemon Blue DQN/communication/reset.txt'
local exit_path = 'C:/Users/jerem/Pokemon Blue DQN/communication/exit.txt'

local action_ready_path = 'C:/Users/jerem/Pokemon Blue DQN/communication/action_ready.txt'
local state_ready_path = 'C:/Users/jerem/Pokemon Blue DQN/communication/state_ready.txt'
local save_slot = 1

local DEBUG_MODE = false
local function sleep_ms(ms)
    local start = os.clock()
    while os.clock() - start < (ms / 1000) do end
end

local memory_locations = {}

table.insert(memory_locations, 0xD014) -- pokemon #1 number
table.insert(memory_locations, 0xD015) -- pokemon #1 health (byte 1)
table.insert(memory_locations, 0xD016) -- pokemon #1 health (byte 2)
table.insert(memory_locations, 0xD023) -- pokemon #1 health 1
table.insert(memory_locations, 0xD024) -- pokemon #1 health 2
table.insert(memory_locations, 0xD022) -- pokemon #1 level
table.insert(memory_locations, 0xD179) -- pokemon #1 exp 1
table.insert(memory_locations, 0xD17A) -- pokemon #1 exp 2
table.insert(memory_locations, 0xD17B) -- pokemon #1 exp 3
table.insert(memory_locations, 0xD019) -- pokemon #1 type 1
table.insert(memory_locations, 0xD01A) -- pokemon #1 type 2

table.insert(memory_locations, 0xD01C) -- player pokemon #1 move #1
table.insert(memory_locations, 0xD01D) -- player pokemon #1 move #2
table.insert(memory_locations, 0xD01E) -- player pokemon #1 move #3
table.insert(memory_locations, 0xD01F) -- player pokemon #1 move #4
table.insert(memory_locations, 0xD02D) -- player pokemon #1 move #1 pp
table.insert(memory_locations, 0xD02E) -- player pokemon #1 move #2 pp
table.insert(memory_locations, 0xD02F) -- player pokemon #1 move #3 pp
table.insert(memory_locations, 0xD030) -- player pokemon #1 move #4 pp

table.insert(memory_locations, 0xCCDC) -- player-selected move
table.insert(memory_locations, 0xCC1A) -- player pokemon attack mod (7 means no mod)
table.insert(memory_locations, 0xCC1B) -- player pokemon defense mod
table.insert(memory_locations, 0xCC1C) -- player pokemon speed mod
table.insert(memory_locations, 0xCC1D) -- player pokemon special mod
table.insert(memory_locations, 0xCC1E) -- player pokemon accuracy mod
table.insert(memory_locations, 0xCC1F) -- player pokemon evasion mod
table.insert(memory_locations, 0xD018) -- player pokemon #1 status

table.insert(memory_locations, 0xCFD8) -- enemy pokemon internal ID
table.insert(memory_locations, 0xCFE6) -- enemy pokemon hp 1
table.insert(memory_locations, 0xCFE7) -- enemy pokemon hp 2
table.insert(memory_locations, 0xCFF4) -- enemy pokemon max hp 1
table.insert(memory_locations, 0xCFF5) -- enemy pokemon max hp 2
table.insert(memory_locations, 0xCCDD) -- enemy-selected move
table.insert(memory_locations, 0xCC2F) -- enemy pokemon defense mod
table.insert(memory_locations, 0xCC30) -- enemy pokemon speed mod
table.insert(memory_locations, 0xCC31) -- enemy pokemon special mod
table.insert(memory_locations, 0xCC32) -- enemy pokemon accuracy mod
table.insert(memory_locations, 0xCC33) -- enemy pokemon evasion mod
table.insert(memory_locations, 0xCFE9) -- enemy pokemon status

table.insert(memory_locations, 0xD057) -- type of battle
table.insert(memory_locations, 0xCCD5) -- number of turns in battle

table.insert(memory_locations, 0xD163) -- number of pokemon in party
table.insert(memory_locations, 0xD198) -- player pokemon 2 hp 1
table.insert(memory_locations, 0xD199) -- player pokemon 2 hp 2
table.insert(memory_locations, 0xD1C4) -- player pokemon 3 hp 1
table.insert(memory_locations, 0xD1C5) -- player pokemon 3 hp 2
table.insert(memory_locations, 0xD1F0) -- player pokemon 4 hp 1
table.insert(memory_locations, 0xD1F1) -- player pokemon 4 hp 2
table.insert(memory_locations, 0xD21C) -- player pokemon 5 hp 1
table.insert(memory_locations, 0xD21D) -- player pokemon 5 hp 2
table.insert(memory_locations, 0xD248) -- player pokemon 6 hp 1
table.insert(memory_locations, 0xD249) -- player pokemon 6 hp 2

table.insert(memory_locations, 0xD347) -- money (byte 1)
table.insert(memory_locations, 0xD348) -- money (byte 2)
table.insert(memory_locations, 0xD349) -- money (byte 3)
table.insert(memory_locations, 0xD31D) -- number of items
table.insert(memory_locations, 0xD356) -- number of badges

table.insert(memory_locations, 0xD35E) -- current map number
table.insert(memory_locations, 0xD362) -- x-position of player
table.insert(memory_locations, 0xD361) -- y-position of player

table.insert(memory_locations, 0xCC25) -- cursor x-position
table.insert(memory_locations, 0xCC24) -- cursor y-position
table.insert(memory_locations, 0xCC26) -- selected menu item
table.insert(memory_locations, 0xCCDB) -- menu type

local function get_gb_memory_values()
    local values = {}
    for _, address in ipairs(memory_locations) do
        local success, val = pcall(emu.read8, emu, address)
        values[#values + 1] = success and val or 0
    end
    return values
end


local total_frames = 0 -- how many frames have elapsed in total

local function check_exit()
    local f = io.open(exit_path, 'r')
    if f then
        f:close()
        os.remove(exit_path)
        return true
    end
    return false
end

local function get_game_state()
    local values = get_gb_memory_values()
    values[#values + 1] = total_frames
    return table.concat(values, ',')
end


local function load_init_state()
    emu:loadStateSlot(save_slot)
end


local function signal_ready_for_action()
    local file = io.open(action_ready_path, 'w')
    if file then
        file:write('1')
        file:close()
    end
end


local function wait_for_action_from_python()
    while true do
        local exit_file = io.open(exit_path, 'r')
        if exit_file then
            exit_file:close()
            os.remove(exit_path)
            return true
        end

        local f = io.open(action_ready_path, 'r')
        if not f then break end
        f:close()
        emu:runFrame()
    end

    return false
end


local ACTION_DURATION = 4 -- how many frames an action will be held for
local MOVEMENT_DELAY = 18 -- how many frames until next action is inputted

local frame_counter = 0 -- how many frames
local action_to_hold = nil -- which action to input
local in_post_action_delay = false

while true do
    if check_exit() then return end

    local reset_file = io.open(reset_path, 'r')
    if reset_file then
        reset_file:close()
        os.remove(reset_path)
        load_init_state()
        total_frames = 0
    end

    if not action_to_hold and not in_post_action_delay then
        signal_ready_for_action()
        if wait_for_action_from_python() then return end

        local action_file = io.open(action_path, 'r')
        if action_file then
            local action = tonumber(action_file:read('*l'))
            action_file:close()
            os.remove(action_path)

            if action then
                action_to_hold = action
                frame_counter = 0
                emu:addKey(action)
            end
        end
    elseif action_to_hold and frame_counter >= ACTION_DURATION then
        emu:clearKey(action_to_hold)
        action_to_hold = nil
        in_post_action_delay = true
        frame_counter = 0
    elseif in_post_action_delay and frame_counter >= MOVEMENT_DELAY then
        local state_file = io.open(state_path, 'w')
        if state_file then
            state_file:write(get_game_state())
            state_file:close()
        end

        local ready_file = io.open(state_ready_path, 'w')
        if ready_file then
            ready_file:write('1')
            ready_file:close()
        end

        in_post_action_delay = false
        frame_counter = 0
    end

    if check_exit() then return end

    emu:runFrame()
    if DEBUG_MODE then sleep_ms(10) end
    total_frames = total_frames + 1
    frame_counter = frame_counter + 1
end