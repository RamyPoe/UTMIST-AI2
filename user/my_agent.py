# # SUBMISSION: Agent
# This will be the Agent class we run in the 1v1. We've started you off with a functioning RL agent (`SB3Agent(Agent)`) and if-statement agent (`BasedAgent(Agent)`). Feel free to copy either to `SubmittedAgent(Agent)` then begin modifying.
# 
# Requirements:
# - Your submission **MUST** be of type `SubmittedAgent(Agent)`
# - Any instantiated classes **MUST** be defined within and below this code block.
# 
# Remember, your agent can be either machine learning, OR if-statement based. I've seen many successful agents arising purely from if-statements - give them a shot as well, if ML is too complicated at first!!
# 
# Also PLEASE ask us questions in the Discord server if any of the API is confusing. We'd be more than happy to clarify and get the team on the right track.
# Requirements:
# - **DO NOT** import any modules beyond the following code block. They will not be parsed and may cause your submission to fail validation.
# - Only write imports that have not been used above this code block
# - Only write imports that are from libraries listed here
# We're using PPO by default, but feel free to experiment with other Stable-Baselines 3 algorithms!

import os
import gdown
from typing import Optional
from environment.agent import Agent
from stable_baselines3 import PPO, A2C # Sample RL Algo imports
from sb3_contrib import RecurrentPPO # Importing an LSTM

# To run the sample TTNN model, you can uncomment the 2 lines below: 
# import ttnn
# from user.my_agent_tt import TTMLPPolicy

class MoveType():
    NONE = 1          # no move
    NLIGHT = 2        # grounded light neutral
    DLIGHT = 3        # grounded light down
    SLIGHT = 4        # grounded light side
    NSIG = 5          # grounded heavy neutral
    DSIG = 6          # grounded heavy down
    SSIG = 7          # grounded heavy side
    NAIR = 8          # aerial light neutral
    DAIR = 9          # aerial light down
    SAIR = 10         # aerial light side
    RECOVERY = 11     # aerial heavy neutral and aerial heavy side
    GROUNDPOUND = 12  # aerial heavy down
    
    @staticmethod
    def isUp(move):
        return move in [MoveType.NLIGHT, MoveType.NSIG, MoveType.NAIR, MoveType.RECOVERY]

    @staticmethod
    def isDown(move):
        return move in [MoveType.DLIGHT, MoveType.DSIG, MoveType.DAIR, MoveType.GROUNDPOUND]
    
    @staticmethod
    def isSide(move):
        return move in [MoveType.SLIGHT, MoveType.SSIG, MoveType.SAIR]
    
    @staticmethod
    def isLight(move):
        return move in [MoveType.NLIGHT, MoveType.DLIGHT, MoveType.SLIGHT, MoveType.NAIR, MoveType.DAIR, MoveType.SAIR]
    
    @staticmethod
    def isHeavy(move):
        return move in [MoveType.NSIG, MoveType.DSIG, MoveType.SSIG, MoveType.RECOVERY, MoveType.GROUNDPOUND]
        
class WeaponType():
    UNARMED = 0
    SPEAR = 1
    HAMMER = 2

class SubmittedAgent(Agent):
    '''
    Input the **file_path** to your agent here for submission!
    '''
    def __init__(
        self,
        file_path: Optional[str] = None,
    ):
        super().__init__(file_path)
        self.unpressAllKeys()
        self.time = 0
        self.FRAME_TIME = 1 / 30.0
        self.MAINTAIN_DISTANCE = 2
        self.PREDICTION_DAMP = 0.5
              
        self.attackDelayMap = {
            WeaponType.UNARMED: {
                MoveType.NLIGHT : 3,
                MoveType.DLIGHT : 5,
                MoveType.SLIGHT : 6,
                MoveType.NAIR : 4,
                MoveType.DAIR : 6,
                MoveType.SAIR : 6,
                MoveType.NSIG : 10,
                MoveType.DSIG : 11,
                MoveType.SSIG : 11,
                MoveType.RECOVERY : 6,
                MoveType.GROUNDPOUND : 10
            },
            WeaponType.HAMMER: {
                MoveType.NLIGHT : 4,
                MoveType.DLIGHT : 5,
                MoveType.SLIGHT : 7,
                MoveType.NAIR : 9,
                MoveType.DAIR : 6,
                MoveType.SAIR : 7,
                MoveType.NSIG : 9,
                MoveType.DSIG : 16,
                MoveType.SSIG : 17,
                MoveType.RECOVERY : 7,
                MoveType.GROUNDPOUND : 13
            },
            WeaponType.SPEAR: {
                MoveType.NLIGHT : 4,
                MoveType.DLIGHT : 5,
                MoveType.SLIGHT : 7,
                MoveType.NAIR : 8,
                MoveType.DAIR : 6,
                MoveType.SAIR : 7,
                MoveType.NSIG : 11,
                MoveType.DSIG : 19,
                MoveType.SSIG : 16,
                MoveType.RECOVERY : 8,
                MoveType.GROUNDPOUND : 9
            }
        }
        
        # Hitbox Map (xOffset, yOffset, width, height)
        self.attackHitboxMap = {
            WeaponType.UNARMED: {
                True: {
                    MoveType.NAIR : (0.6, 0, 0.5, 0.8),
                    MoveType.DAIR : (1, 1, 0.5, 0.5)
                },
                False: {
                    MoveType.DSIG : (0.2, 0.25, 1.7, 0.5),
                    MoveType.NLIGHT : (0.5, 0, 0.5, 0.8)
                }
            },
            WeaponType.SPEAR: {
                True: {
                    MoveType.NAIR : (0.3, 0, 1.7, 1.8),
                    MoveType.SAIR : (1.6, 0.15, 0.4, 0.8)
                },
                False: {
                    MoveType.NSIG : (0.6, -0.8, 2, 0.7),
                    MoveType.SLIGHT : (1, 0, 1.3, 0.9)
                }
            },
            WeaponType.HAMMER: {
                True: {
                    MoveType.SAIR : (1.1, 0, 0.7, 0.7),
                    MoveType.NAIR : (0.4, -1.3, 0.9, 0.9)
                },
                False: {
                    MoveType.NSIG : (0.8, -0.6, 0.4, 1.2),
                    MoveType.SLIGHT : (1.3, -0.25, 1.2, 1.25)
                }
            }
        }

        
        self.count = 0
        
        # Prediction States
        self.oppNextAttack = -1
        self.oppLastState = None


    def printDebugAction(self):
        # 5 - weapon pickup/drop
        # 6 - dash/dodge
        # 7 - light attack
        # 8 - heavy attack
        # 9 - emote
        #                0    1    2    3      4      5    6    7    8    9
        action_names = ["W", "A", "S", "D", "space", 'h', 'l', 'j', 'k', 'g']
        pressed = [action_names[i] for i in range(len(self.action)) if self.action[i] == 1]
        print(f"Time {self.time}: Pressing {pressed}")

    def moveLeft(self):
        self.action[1] = 1
        self.action[3] = 0

    def moveRight(self):
        self.action[1] = 0
        self.action[3] = 1

    def stopMoveHorizontal(self):
        self.action[1] = 0
        self.action[3] = 0
        
    def stopMoveVertical(self):
        self.action[0] = 0
        self.action[2] = 0
        
    def holdUp(self):
        self.action[0] = 1
        self.action[2] = 0

    def holdDown(self):
        self.action[0] = 0
        self.action[2] = 1
        
    def setJump(self, jump=True):
        self.action[4] = 1 if jump else 0

    def setDodgeDash(self, dodge=True):
        self.action[6] = 1 if dodge else 0

    def setPickupDrop(self, pickup=True):
        self.action[5] = 1 if pickup else 0

    def setLightAttack(self, attack=True):
        if not attack: return
        self.action[7] = 1
        self.action[8] = 0

    def setHeavyAttack(self, attack=True):
        if not attack: return
        self.action[8] = 1
        self.action[7] = 0
        
    def stopAttacking(self):
        self.action[7] = 0
        self.action[8] = 0
    
    def unpressAllKeys(self):
        self.action = [0 for _ in range(10)]
    
    @staticmethod
    def isMiddleGap(pos):
        return -2.5 <= pos[0] <= 2.5
        
    @staticmethod
    def isOffRightSide(pos):
        return pos[0] > 6
        
    @staticmethod
    def isOffLeftSide(pos):
        return pos[0] < -6

    @staticmethod
    def isTowardsRightSide(pos):
        return pos[0] > 4
        
    @staticmethod
    def isTowardsLeftSide(pos):
        return pos[0] < -4
        
    @staticmethod
    def belowLowerStage(pos):
        return pos[1] > 2.32

    @staticmethod
    def belowUpperStage(pos):
        return pos[1] > 0.4
    
    def moveJumpRecover(self, jumpsLeft, inAir, pos, vel):
        if vel[1] < -1.8 or pos[1] < -3: return # Still rising or too high
        self.stopAttacking()
        if not inAir or jumpsLeft != 0:
            self.setJump(self.time % 2 == 0) # Tap jump
        elif inAir:
            self.holdUp()
            self.setHeavyAttack()
            
    # Assume platform is centered at platPos and 1.8 units wide
    # Assume myPos is centered at player and 0.928 unit wide
    @staticmethod
    def isVerticallyLinedWithPlatform(myPos, platPos):
        return (myPos[0] + 0.4 >= platPos[0] - 0.9) and (myPos[0] - 0.4 <= platPos[0] + 0.9)
    
    @staticmethod
    def isRightEndOfPlatform(myPos, platPos):
        return myPos[0] > platPos[0] + 0.3 and SubmittedAgent.isVerticallyLinedWithPlatform(myPos, platPos)

    @staticmethod
    def isLeftEndOfPlatform(myPos, platPos):
        return myPos[0] < platPos[0] - 0.3 and SubmittedAgent.isVerticallyLinedWithPlatform(myPos, platPos)
    
    @staticmethod
    def isAbovePlatform(myPos, platPos):
        return myPos[1] < platPos[1]
    
    @staticmethod
    def euclideanDistance(pos1, pos2):
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    
    @staticmethod
    def check_collision(myPos, facingRight, hitbox, oppPos):
        m = 1 if facingRight else -1
        attackBox = [myPos[0] + hitbox[0] * m, myPos[1] + hitbox[1], hitbox[2], hitbox[3]]
        oppBox = [oppPos[0], oppPos[1], 0.928, 1.024]
        
        # AABB collision detection
        return attackBox[0] + attackBox[2] / 2 >= oppBox[0] - oppBox[2] / 2 and \
                attackBox[0] - attackBox[2] / 2 <= oppBox[0] + oppBox[2] / 2 and \
                attackBox[1] + attackBox[3] / 2 >= oppBox[1] - oppBox[3] / 2 and \
                attackBox[1] - attackBox[3] / 2 <= oppBox[1] + oppBox[3] / 2
    
    """
    CHECKLIST:
        - [X] Dodge with frame delay after predicting opponent attack
        - [X] ATTACK THE OPPONENT!
            - [-] If opponent attacks facing away on same y-level, approach and attack
            - [X] If opponent attacks and we dodge (on the ground?), approach and attack
        - [-] Move farther from opponent when no dodge cooldown
        - [X] If closer to left platform, prioritize over center platform
    """
    

    # Logic should start from low priority to high priority
    def predict(self, obs):
        self.time += 1
        self.unpressAllKeys()
        
        # Gather state information
        myPos = self.obs_helper.get_section(obs, 'player_pos')
        myVel = self.obs_helper.get_section(obs, 'player_vel')
        myInAir = int(self.obs_helper.get_section(obs, 'player_aerial')) == 1
        myJumps = self.obs_helper.get_section(obs, 'player_jumps_left')
        myWeapon = int(self.obs_helper.get_section(obs, 'player_weapon_type'))
        myState = self.obs_helper.get_section(obs, 'player_state')
        myFacing = int(self.obs_helper.get_section(obs, 'player_facing')) == 1
        
        oppPos = self.obs_helper.get_section(obs, 'opponent_pos')
        oppVel = self.obs_helper.get_section(obs, 'opponent_vel')
        oppState = self.obs_helper.get_section(obs, 'opponent_state')
        oppMove = int(self.obs_helper.get_section(obs, 'opponent_move_type'))
        oppWeapon = int(self.obs_helper.get_section(obs, 'opponent_weapon_type'))
        
        platPos = self.obs_helper.get_section(obs, 'player_moving_platform_pos')
        platVel = self.obs_helper.get_section(obs, 'player_moving_platform_vel')
        spawners = [self.obs_helper.get_section(obs, f'player_spawner_{i+1}') for i in range(4)]
        
        
        
        # Determine goal position
        goalX = None
        
        if (myWeapon == WeaponType.UNARMED): # Unarmed, go for weapon
            closest = None
            for spawner in spawners:
                if not spawner[2]: continue # No weapon present
                distToMe = abs(spawner[0] - myPos[0])
                if not closest or distToMe < closest:
                    goalX, closest = spawner[0], distToMe

            if closest and closest < 0.8:
                goalX = None # Already close enough to weapon to pick up
                self.setPickupDrop()

        # Stay within attacking range of opponent
        if not goalX:
            # Go through if opponent edge gaurding right side
            if self.isTowardsRightSide(oppPos) and myPos[0] > oppPos[0]:
                goalX = oppPos[0] - self.MAINTAIN_DISTANCE
            # Go through if opponent edge gaurding left side
            elif self.isTowardsLeftSide(oppPos) and myPos[0] < oppPos[0]:
                goalX = oppPos[0] + self.MAINTAIN_DISTANCE
            else:
                goalX = oppPos[0] + (-self.MAINTAIN_DISTANCE if myPos[0] < oppPos[0] else self.MAINTAIN_DISTANCE)
                 

        # Basically made it here, so just idle
        if goalX and abs(goalX - myPos[0]) < 0.5:
            goalX = None
        
        # Move towards goal
        if goalX:
            if goalX > myPos[0]:
                self.moveRight()
            else:
                self.moveLeft()



        # Attack if hitbox collides opponent
        attacks = self.attackHitboxMap[myWeapon][myInAir]
        for move, hitBox in attacks.items():
            # break
            # Predict opponent position
            oppPosFuture = [
                oppPos[0] + self.PREDICTION_DAMP * oppVel[0] * self.FRAME_TIME * self.attackDelayMap[myWeapon][move],
                oppPos[1] + self.PREDICTION_DAMP * oppVel[1] * self.FRAME_TIME * self.attackDelayMap[myWeapon][move]
            ]
            
            # Check future collision
            attackPrimed = self.check_collision(myPos, myFacing, hitBox, oppPos)
            if not attackPrimed: continue
            if MoveType.isUp(move):
                self.holdUp()
                self.stopMoveHorizontal()
            elif MoveType.isDown(move):
                self.holdDown()
                self.stopMoveHorizontal()
            elif MoveType.isSide(move):
                self.moveRight() if oppPos[0] > myPos[0] else self.moveLeft()
                self.stopMoveVertical()
            if MoveType.isLight(move):
                self.setLightAttack()
            elif MoveType.isHeavy(move):
                self.setHeavyAttack(self.time % 2 == 0)
            break
        else:
            # If we have no chances, check if turning around would help next frame
            for move, hitBox in attacks.items():
                attackPrimed = self.check_collision(myPos, not myFacing, hitBox, oppPos) # TODO: check future?
                if not attackPrimed: continue
                self.moveLeft() if myFacing else self.moveRight()
                break
        
        
        # If off the edge, come back
        if self.isOffRightSide(myPos):
            self.moveLeft()
            self.moveJumpRecover(myJumps, myInAir, myPos, myVel)
        elif self.isOffLeftSide(myPos):
            self.moveRight()
            self.moveJumpRecover(myJumps, myInAir, myPos, myVel)
        elif self.isMiddleGap(myPos) and myInAir:
            # If in middle gap, try to recover to platform
            if not self.isAbovePlatform(myPos, platPos):
                self.stopAttacking()
                prioritizeLeftPlatform = myJumps == 0 or (myJumps == 1 and myPos[1] - platPos[1] > 1)
                
                if self.isVerticallyLinedWithPlatform(myPos, platPos):
                    if myPos[1] - platPos[1] > 2 or prioritizeLeftPlatform:
                        self.moveJumpRecover(myJumps, myInAir, myPos, myVel)

                    # Platform on right side, go around left because gap is too small
                    if platPos[0] > 0.6 - (0.6 if platVel[0] > 0 else 0):
                        self.moveLeft()
                    else:
                        if myPos[0] < platPos[0]: self.moveLeft()
                        else: self.moveRight()
                else:
                    self.moveJumpRecover(myJumps, myInAir, myPos, myVel)
                    
                # Under the platform, go for the left side
                if prioritizeLeftPlatform:
                    self.moveLeft()
            else:
                if goalX and myJumps > 1: # Prioritize going towards goal
                    if goalX > myPos[0] and myVel[0] > 0: self.moveRight()
                    elif goalX < myPos[0] and myVel[0] < 0: self.moveLeft()
                else:  # Go towards platform
                    if myPos[0] < platPos[0]: self.moveRight()
                    else: self.moveLeft()
        elif self.isMiddleGap(myPos) and not myInAir: # Momentum jump from inner ledges
            if (myPos[0] < -2 and myVel[0] > 0) or \
               (myPos[0] > 2 and myVel[0] < 0)  or \
               (self.isRightEndOfPlatform(myPos, platPos) and myVel[0] > 0) or \
               (self.isLeftEndOfPlatform(myPos, platPos) and myVel[0] < 0):
                self.moveJumpRecover(myJumps, myInAir, myPos, myVel)
        
        
        # Predict opponent attack with frame delay and dodge
        if oppState == 8 and self.oppLastState != 8:
            self.oppNextAttack = self.time + self.attackDelayMap[oppWeapon][oppMove] - 2
        if self.time == self.oppNextAttack and self.euclideanDistance(myPos, oppPos) < 2.5:
            self.stopMoveHorizontal()
            self.stopAttacking()
            self.setDodgeDash()
            self.oppNextAttack = -1


        # Never drop weapon
        if myWeapon != 0:
            self.setPickupDrop(pickup=False)

        
        # print(f"Time {self.time}: Jumps -> {myJumps}, PriLeft -> {myJumps == 0 or (myJumps == 1 and myPos[1] - platPos[1] > 1)}, VertAligned -> {self.isVerticallyLinedWithPlatform(myPos, platPos)}")
        
        # print(f"Time {self.time}: {type(oppMove)} {oppMove} {int(oppMove)}")
        # self.stopMoveHorizontal()

        # self.unpressAllKeys()
        # print(f"Time {self.time}: {oppMove}")
        # print(f"Time {self.time}: {oppState} {myState}")
        
        # print(f"Time {self.time}: GoalX {goalX}")
        
        # if oppState != 8 or myState == 5:
        #     self.count = 0
        # else:
        #     self.count += 1
        
        # if (self.count != 0):
        #     print(f"Time {self.time}: Count {self.count}")

        # print(f"Time {self.time}: Pos {oppPos}, Vel {oppVel}")
        # print(f"Time {self.time}: Pos {myPos}, Vel {myVel}, InAir {myInAir}, JumpsLeft {myJumps}")
        # print(f"Time {self.time}: {platPos[1] - myPos[1]}")
        
        self.oppLastState = oppState
        return self.action
