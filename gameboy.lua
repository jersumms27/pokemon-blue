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

-- MAP AND PLAYER LOCATION
table.insert(memory_locations, 0xD35E) -- current map number
table.insert(memory_locations, 0xD362) -- x-position of player
table.insert(memory_locations, 0xD361) -- y-position of player
table.insert(memory_locations, 0xD363) -- current player Y-position (current block)
table.insert(memory_locations, 0xD364) -- current player X-position (current block)

-- POKEMON #1 (ACTIVE POKEMON) - COMPLETE DATA
table.insert(memory_locations, 0xD014) -- pokemon #1 number
table.insert(memory_locations, 0xD015) -- pokemon #1 current hp (byte 1)
table.insert(memory_locations, 0xD016) -- pokemon #1 current hp (byte 2)
table.insert(memory_locations, 0xD023) -- pokemon #1 max hp 1
table.insert(memory_locations, 0xD024) -- pokemon #1 max hp 2
table.insert(memory_locations, 0xD022) -- pokemon #1 level
table.insert(memory_locations, 0xD179) -- pokemon #1 exp 1
table.insert(memory_locations, 0xD17A) -- pokemon #1 exp 2
table.insert(memory_locations, 0xD17B) -- pokemon #1 exp 3
table.insert(memory_locations, 0xD019) -- pokemon #1 type 1
table.insert(memory_locations, 0xD01A) -- pokemon #1 type 2
table.insert(memory_locations, 0xD01C) -- pokemon #1 move #1
table.insert(memory_locations, 0xD01D) -- pokemon #1 move #2
table.insert(memory_locations, 0xD01E) -- pokemon #1 move #3
table.insert(memory_locations, 0xD01F) -- pokemon #1 move #4
table.insert(memory_locations, 0xD02D) -- pokemon #1 move #1 pp
table.insert(memory_locations, 0xD02E) -- pokemon #1 move #2 pp
table.insert(memory_locations, 0xD02F) -- pokemon #1 move #3 pp
table.insert(memory_locations, 0xD030) -- pokemon #1 move #4 pp
table.insert(memory_locations, 0xD018) -- pokemon #1 status
table.insert(memory_locations, 0xD025) -- pokemon #1 attack 1
table.insert(memory_locations, 0xD026) -- pokemon #1 attack 2
table.insert(memory_locations, 0xD027) -- pokemon #1 defense 1
table.insert(memory_locations, 0xD028) -- pokemon #1 defense 2
table.insert(memory_locations, 0xD029) -- pokemon #1 speed 1
table.insert(memory_locations, 0xD02A) -- pokemon #1 speed 2
table.insert(memory_locations, 0xD02B) -- pokemon #1 special 1
table.insert(memory_locations, 0xD02C) -- pokemon #1 special 2

-- POKEMON #2 - COMPLETE DATA
table.insert(memory_locations, 0xD197) -- pokemon #2 number
table.insert(memory_locations, 0xD198) -- pokemon #2 current hp 1
table.insert(memory_locations, 0xD199) -- pokemon #2 current hp 2
table.insert(memory_locations, 0xD1B9) -- pokemon #2 max hp 1
table.insert(memory_locations, 0xD1BA) -- pokemon #2 max hp 2
table.insert(memory_locations, 0xD1B8) -- pokemon #2 level
table.insert(memory_locations, 0xD1A5) -- pokemon #2 exp 1
table.insert(memory_locations, 0xD1A6) -- pokemon #2 exp 2
table.insert(memory_locations, 0xD1A7) -- pokemon #2 exp 3
table.insert(memory_locations, 0xD19C) -- pokemon #2 type 1
table.insert(memory_locations, 0xD19D) -- pokemon #2 type 2
table.insert(memory_locations, 0xD19F) -- pokemon #2 move #1
table.insert(memory_locations, 0xD1A0) -- pokemon #2 move #2
table.insert(memory_locations, 0xD1A1) -- pokemon #2 move #3
table.insert(memory_locations, 0xD1A2) -- pokemon #2 move #4
table.insert(memory_locations, 0xD1B4) -- pokemon #2 move #1 pp
table.insert(memory_locations, 0xD1B5) -- pokemon #2 move #2 pp
table.insert(memory_locations, 0xD1B6) -- pokemon #2 move #3 pp
table.insert(memory_locations, 0xD1B7) -- pokemon #2 move #4 pp
table.insert(memory_locations, 0xD19B) -- pokemon #2 status

-- POKEMON #3 - COMPLETE DATA
table.insert(memory_locations, 0xD1C3) -- pokemon #3 number
table.insert(memory_locations, 0xD1C4) -- pokemon #3 current hp 1
table.insert(memory_locations, 0xD1C5) -- pokemon #3 current hp 2
table.insert(memory_locations, 0xD1E5) -- pokemon #3 max hp 1
table.insert(memory_locations, 0xD1E6) -- pokemon #3 max hp 2
table.insert(memory_locations, 0xD1E4) -- pokemon #3 level
table.insert(memory_locations, 0xD1D1) -- pokemon #3 exp 1
table.insert(memory_locations, 0xD1D2) -- pokemon #3 exp 2
table.insert(memory_locations, 0xD1D3) -- pokemon #3 exp 3
table.insert(memory_locations, 0xD1C8) -- pokemon #3 type 1
table.insert(memory_locations, 0xD1C9) -- pokemon #3 type 2
table.insert(memory_locations, 0xD1CB) -- pokemon #3 move #1
table.insert(memory_locations, 0xD1CC) -- pokemon #3 move #2
table.insert(memory_locations, 0xD1CD) -- pokemon #3 move #3
table.insert(memory_locations, 0xD1CE) -- pokemon #3 move #4
table.insert(memory_locations, 0xD1E0) -- pokemon #3 move #1 pp
table.insert(memory_locations, 0xD1E1) -- pokemon #3 move #2 pp
table.insert(memory_locations, 0xD1E2) -- pokemon #3 move #3 pp
table.insert(memory_locations, 0xD1E3) -- pokemon #3 move #4 pp
table.insert(memory_locations, 0xD1C7) -- pokemon #3 status

-- POKEMON #4 - COMPLETE DATA
table.insert(memory_locations, 0xD1EF) -- pokemon #4 number
table.insert(memory_locations, 0xD1F0) -- pokemon #4 current hp 1
table.insert(memory_locations, 0xD1F1) -- pokemon #4 current hp 2
table.insert(memory_locations, 0xD211) -- pokemon #4 max hp 1
table.insert(memory_locations, 0xD212) -- pokemon #4 max hp 2
table.insert(memory_locations, 0xD210) -- pokemon #4 level
table.insert(memory_locations, 0xD1FD) -- pokemon #4 exp 1
table.insert(memory_locations, 0xD1FE) -- pokemon #4 exp 2
table.insert(memory_locations, 0xD1FF) -- pokemon #4 exp 3
table.insert(memory_locations, 0xD1F4) -- pokemon #4 type 1
table.insert(memory_locations, 0xD1F5) -- pokemon #4 type 2
table.insert(memory_locations, 0xD1F7) -- pokemon #4 move #1
table.insert(memory_locations, 0xD1F8) -- pokemon #4 move #2
table.insert(memory_locations, 0xD1F9) -- pokemon #4 move #3
table.insert(memory_locations, 0xD1FA) -- pokemon #4 move #4
table.insert(memory_locations, 0xD20C) -- pokemon #4 move #1 pp
table.insert(memory_locations, 0xD20D) -- pokemon #4 move #2 pp
table.insert(memory_locations, 0xD20E) -- pokemon #4 move #3 pp
table.insert(memory_locations, 0xD20F) -- pokemon #4 move #4 pp
table.insert(memory_locations, 0xD1F3) -- pokemon #4 status

-- POKEMON #5 - COMPLETE DATA
table.insert(memory_locations, 0xD21B) -- pokemon #5 number
table.insert(memory_locations, 0xD21C) -- pokemon #5 current hp 1
table.insert(memory_locations, 0xD21D) -- pokemon #5 current hp 2
table.insert(memory_locations, 0xD23D) -- pokemon #5 max hp 1
table.insert(memory_locations, 0xD23E) -- pokemon #5 max hp 2
table.insert(memory_locations, 0xD23C) -- pokemon #5 level
table.insert(memory_locations, 0xD229) -- pokemon #5 exp 1
table.insert(memory_locations, 0xD22A) -- pokemon #5 exp 2
table.insert(memory_locations, 0xD22B) -- pokemon #5 exp 3
table.insert(memory_locations, 0xD220) -- pokemon #5 type 1
table.insert(memory_locations, 0xD221) -- pokemon #5 type 2
table.insert(memory_locations, 0xD223) -- pokemon #5 move #1
table.insert(memory_locations, 0xD224) -- pokemon #5 move #2
table.insert(memory_locations, 0xD225) -- pokemon #5 move #3
table.insert(memory_locations, 0xD226) -- pokemon #5 move #4
table.insert(memory_locations, 0xD238) -- pokemon #5 move #1 pp
table.insert(memory_locations, 0xD239) -- pokemon #5 move #2 pp
table.insert(memory_locations, 0xD23A) -- pokemon #5 move #3 pp
table.insert(memory_locations, 0xD23B) -- pokemon #5 move #4 pp
table.insert(memory_locations, 0xD21F) -- pokemon #5 status

-- POKEMON #6 - COMPLETE DATA
table.insert(memory_locations, 0xD247) -- pokemon #6 number
table.insert(memory_locations, 0xD248) -- pokemon #6 current hp 1
table.insert(memory_locations, 0xD249) -- pokemon #6 current hp 2
table.insert(memory_locations, 0xD269) -- pokemon #6 max hp 1
table.insert(memory_locations, 0xD26A) -- pokemon #6 max hp 2
table.insert(memory_locations, 0xD268) -- pokemon #6 level
table.insert(memory_locations, 0xD255) -- pokemon #6 exp 1
table.insert(memory_locations, 0xD256) -- pokemon #6 exp 2
table.insert(memory_locations, 0xD257) -- pokemon #6 exp 3
table.insert(memory_locations, 0xD24C) -- pokemon #6 type 1
table.insert(memory_locations, 0xD24D) -- pokemon #6 type 2
table.insert(memory_locations, 0xD24F) -- pokemon #6 move #1
table.insert(memory_locations, 0xD250) -- pokemon #6 move #2
table.insert(memory_locations, 0xD251) -- pokemon #6 move #3
table.insert(memory_locations, 0xD252) -- pokemon #6 move #4
table.insert(memory_locations, 0xD264) -- pokemon #6 move #1 pp
table.insert(memory_locations, 0xD265) -- pokemon #6 move #2 pp
table.insert(memory_locations, 0xD266) -- pokemon #6 move #3 pp
table.insert(memory_locations, 0xD267) -- pokemon #6 move #4 pp
table.insert(memory_locations, 0xD24B) -- pokemon #6 status

-- PARTY INFORMATION
table.insert(memory_locations, 0xD163) -- number of pokemon in party

-- ENEMY POKEMON - COMPLETE DATA
table.insert(memory_locations, 0xCFD8) -- enemy pokemon internal ID
table.insert(memory_locations, 0xCFE6) -- enemy pokemon current hp 1
table.insert(memory_locations, 0xCFE7) -- enemy pokemon current hp 2
table.insert(memory_locations, 0xCFF4) -- enemy pokemon max hp 1
table.insert(memory_locations, 0xCFF5) -- enemy pokemon max hp 2
table.insert(memory_locations, 0xCFE8) -- enemy pokemon level
table.insert(memory_locations, 0xCFEA) -- enemy pokemon type 1
table.insert(memory_locations, 0xCFEB) -- enemy pokemon type 2
table.insert(memory_locations, 0xCFED) -- enemy pokemon move 1
table.insert(memory_locations, 0xCFEE) -- enemy pokemon move 2
table.insert(memory_locations, 0xCFEF) -- enemy pokemon move 3
table.insert(memory_locations, 0xCFF0) -- enemy pokemon move 4
table.insert(memory_locations, 0xCFFE) -- enemy pokemon pp (first slot)
table.insert(memory_locations, 0xCFFF) -- enemy pokemon pp (second slot)
table.insert(memory_locations, 0xD000) -- enemy pokemon pp (third slot)
table.insert(memory_locations, 0xD001) -- enemy pokemon pp (fourth slot)
table.insert(memory_locations, 0xCFE9) -- enemy pokemon status
table.insert(memory_locations, 0xCFF6) -- enemy pokemon attack 1
table.insert(memory_locations, 0xCFF7) -- enemy pokemon attack 2
table.insert(memory_locations, 0xCFF8) -- enemy pokemon defense 1
table.insert(memory_locations, 0xCFF9) -- enemy pokemon defense 2
table.insert(memory_locations, 0xCFFA) -- enemy pokemon speed 1
table.insert(memory_locations, 0xCFFB) -- enemy pokemon speed 2
table.insert(memory_locations, 0xCFFC) -- enemy pokemon special 1
table.insert(memory_locations, 0xCFFD) -- enemy pokemon special 2

-- STAT MODIFIERS
table.insert(memory_locations, 0xCD1A) -- player pokemon attack modifier (7 means no modifier)
table.insert(memory_locations, 0xCD1B) -- player pokemon defense modifier
table.insert(memory_locations, 0xCD1C) -- player pokemon speed modifier
table.insert(memory_locations, 0xCD1D) -- player pokemon special modifier
table.insert(memory_locations, 0xCD1E) -- player pokemon accuracy modifier
table.insert(memory_locations, 0xCD1F) -- player pokemon evasion modifier
table.insert(memory_locations, 0xCD2E) -- enemy pokemon attack modifier
table.insert(memory_locations, 0xCD2F) -- enemy pokemon defense modifier
table.insert(memory_locations, 0xCD30) -- enemy pokemon speed modifier
table.insert(memory_locations, 0xCD31) -- enemy pokemon special modifier
table.insert(memory_locations, 0xCD32) -- enemy pokemon accuracy modifier
table.insert(memory_locations, 0xCD33) -- enemy pokemon evasion modifier

-- BATTLE INFORMATION
table.insert(memory_locations, 0xD057) -- type of battle
table.insert(memory_locations, 0xD05A) -- battle type (normal, safari zone, old man battle, etc.)
table.insert(memory_locations, 0xCCD5) -- number of turns in current battle
table.insert(memory_locations, 0xCCDC) -- player-selected move
table.insert(memory_locations, 0xCCDD) -- enemy-selected move
table.insert(memory_locations, 0xCCDB) -- move menu type (0 regular, 1 mimic, other text boxes)
table.insert(memory_locations, 0xD05E) -- critical hit / OHKO flag (01 critical, 02 OHKO)
table.insert(memory_locations, 0xD0D8) -- amount of damage attack is about to do

-- BATTLE STATUS FLAGS (PLAYER)
table.insert(memory_locations, 0xD062) -- battle status player byte 1 (bide, thrash, multi-hit, flinch, etc.)
table.insert(memory_locations, 0xD063) -- battle status player byte 2 (x accuracy, mist, focus energy, substitute, etc.)
table.insert(memory_locations, 0xD064) -- battle status player byte 3 (toxic, light screen, reflect, transformed)

-- BATTLE STATUS FLAGS (ENEMY)
table.insert(memory_locations, 0xD067) -- battle status enemy byte 1
table.insert(memory_locations, 0xD068) -- battle status enemy byte 2
table.insert(memory_locations, 0xD069) -- battle status enemy byte 3

-- BATTLE COUNTERS
table.insert(memory_locations, 0xD06A) -- multi-hit move counter (player)
table.insert(memory_locations, 0xD06B) -- confusion counter (player)
table.insert(memory_locations, 0xD06C) -- toxic counter (player)
table.insert(memory_locations, 0xD06F) -- multi-hit move counter (enemy)
table.insert(memory_locations, 0xD070) -- confusion counter (enemy)
table.insert(memory_locations, 0xD071) -- toxic counter (enemy)

-- MONEY
table.insert(memory_locations, 0xD347) -- money (byte 1)
table.insert(memory_locations, 0xD348) -- money (byte 2)
table.insert(memory_locations, 0xD349) -- money (byte 3)

-- ITEMS
table.insert(memory_locations, 0xD31D) -- total items
table.insert(memory_locations, 0xD31E) -- item 1
table.insert(memory_locations, 0xD31F) -- item 1 quantity
table.insert(memory_locations, 0xD320) -- item 2
table.insert(memory_locations, 0xD321) -- item 2 quantity
table.insert(memory_locations, 0xD322) -- item 3
table.insert(memory_locations, 0xD323) -- item 3 quantity
table.insert(memory_locations, 0xD324) -- item 4
table.insert(memory_locations, 0xD325) -- item 4 quantity
table.insert(memory_locations, 0xD326) -- item 5
table.insert(memory_locations, 0xD327) -- item 5 quantity

-- BADGES
table.insert(memory_locations, 0xD356) -- number of badges

-- CURSOR AND MENU
table.insert(memory_locations, 0xCC25) -- cursor x-position
table.insert(memory_locations, 0xCC24) -- cursor y-position
table.insert(memory_locations, 0xCC26) -- selected menu item (topmost is 0)
table.insert(memory_locations, 0xCC2C) -- last position of cursor on item screen
table.insert(memory_locations, 0xCC2D) -- last position of cursor on START/battle menu
table.insert(memory_locations, 0xCC35) -- item highlighted with Select (01 = first item, 00 = no item)

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