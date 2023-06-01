import numpy as np
from Globals import *
from Drawer import Drawer
from ShapeObjects import *
from addMethods import *
import pygame

drawer = Drawer()
vec2 = pygame.math.Vector2


class Game:
    no_of_actions = 9
    state_size = 20

    def __init__(self):
        trackImg = pyglet.image.load('Track1.png')
        self.trackSprite = pyglet.sprite.Sprite(trackImg, x=0, y=0)

        self.walls = []
        self.gates = []
        self.destination_zone = []

        self.set_walls()
        self.set_gates()
        self.set_destination()

        self.drone = Drone(self.walls, self.gates, self.destination_zone)

    # для швидшого навчання було добавлено додаткові стінки по маршруту

    def set_walls(self):
        self.walls.append(Wall(416, 822, 416, 269))
        self.walls.append(Wall(416, 819, 1473, 819))
        self.walls.append(Wall(416, 272, 1473, 272))
        self.walls.append(Wall(951, 272, 951, 398))
        self.walls.append(Wall(954, 819, 954, 487))
        self.walls.append(Wall(1471, 822, 1471, 268))
        self.walls.append(Wall(630, 360, 750, 395, False))
        self.walls.append(Wall(670, 475, 600, 440, False))
        self.walls.append(Wall(745, 490, 670, 475, False))
        self.walls.append(Wall(745, 490, 790, 490, False))
        self.walls.append(Wall(750, 395, 1070, 405, False))
        self.walls.append(Wall(790, 490, 1070, 490, False))
        self.walls.append(Wall(1070, 405, 1150, 410, False))
        self.walls.append(Wall(1070, 490, 1150, 495, False))
        self.walls.append(Wall(1150, 410, 1250, 460, False))
        self.walls.append(Wall(1150, 495, 1200, 530, False))
        self.walls.append(Wall(1250, 460, 1300, 540, False))
        self.walls.append(Wall(1200, 530, 1210, 620, False))
        self.walls.append(Wall(1300, 540, 1320, 620, False))

    def set_gates(self):

        # self.gates.append(RewardGate(630, 540, 725, 570))
        # self.gates.append(RewardGate(650, 500, 730, 530))
        self.gates.append(RewardGate(730, 380, 700, 475))
        self.gates.append(RewardGate(760, 400, 750, 480))
        self.gates.append(RewardGate(790, 410, 790, 480))
        self.gates.append(RewardGate(830, 410, 830, 480))
        self.gates.append(RewardGate(870, 410, 870, 480))
        self.gates.append(RewardGate(900, 410, 900, 480))
        self.gates.append(RewardGate(930, 410, 930, 480))
        self.gates.append(RewardGate(960, 410, 960, 480))
        self.gates.append(RewardGate(1000, 410, 1000, 480))
        self.gates.append(RewardGate(1040, 410, 1040, 480))
        self.gates.append(RewardGate(1070, 410, 1070, 480))
        self.gates.append(RewardGate(1110, 420, 1090, 483))
        self.gates.append(RewardGate(1150, 430, 1120, 487))
        self.gates.append(RewardGate(1200, 448, 1160, 493))
        self.gates.append(RewardGate(1230, 465, 1180, 505))
        self.gates.append(RewardGate(1250, 490, 1195, 515))
        self.gates.append(RewardGate(1275, 520, 1210, 525))
        self.gates.append(RewardGate(1300, 540, 1220, 540))
        self.gates.append(RewardGate(1300, 560, 1220, 560))
        self.gates.append(RewardGate(1300, 580, 1220, 580))
        self.gates.append(RewardGate(1300, 620, 1220, 620))

    def set_destination(self):
        self.destination_zone.append(DestinationZone(1240, 600, 1240, 640, 1280, 640, 1280, 600))

    def new_episode(self):
        self.drone.reset()

    def get_state(self):
        return self.drone.getState()
        pass

    def make_action(self, action):
        self.drone.updateWithAction(action)
        return self.drone.reward

    def is_episode_finished(self):
        return self.drone.dead

    def render(self):
        glPushMatrix()
        self.trackSprite.draw()

        for w in self.walls:
            w.draw()
        for g in self.gates:
            g.draw()
        for d in self.destination_zone:
            d.draw()
        self.drone.show()
        # self.drone.showCollisionVectors()

        glPopMatrix()


class Wall:

    def __init__(self, x1, y1, x2, y2, lambda1 = True):
        self.x1 = x1
        self.y1 = displayHeight - y1
        self.x2 = x2
        self.y2 = displayHeight - y2

        self.line = Line(self.x1, self.y1, self.x2, self.y2)
        self.line.setLineThinkness(5)
        if lambda1:
            self.line.setColor([255, 0, 0])
        else:
            self.line.setColor([255, 0, 0])
            # self.line.setColor([129,128,129])

    def draw(self):
        self.line.draw()

    #перевіряє на перетин з дроном

    def hitDrone(self, drone):
        global vec2
        rightVector = vec2(drone.direction)
        upVector = vec2(drone.direction).rotate(-90)
        droneCorners = drone.getCorners(rightVector, upVector)
        for i in range(4):
            j = i + 1
            j = j % 4
            if linesCollided(self.x1, self.y1, self.x2, self.y2, droneCorners[i].x, droneCorners[i].y, droneCorners[j].x,
                             droneCorners[j].y):
                return True
        return False

class RewardGate:

    def __init__(self, x1, y1, x2, y2):
        global vec2
        self.x1 = x1
        self.y1 = displayHeight - y1
        self.x2 = x2
        self.y2 = displayHeight - y2
        self.active = True

        self.line = Line(self.x1, self.y1, self.x2, self.y2)
        self.line.setLineThinkness(1)
        self.line.setColor([0, 255, 0])

        self.center = vec2((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def draw(self):
        if self.active:
            self.line.draw()

    #перевіряє на перетин з дроном

    def hitDrone(self, drone):
        if not self.active:
            return False

        global vec2
        rightVector = vec2(drone.direction)
        upVector = vec2(drone.direction).rotate(-90)
        droneCorners = drone.getCorners(rightVector, upVector)
        for i in range(4):
            j = i + 1
            j = j % 4
            if linesCollided(self.x1, self.y1, self.x2, self.y2, droneCorners[i].x, droneCorners[i].y, droneCorners[j].x,
                             droneCorners[j].y):
                return True
        return False


class DestinationZone:

    def __init__(self, x1, y1, x2, y2, x3, y3, x4, y4):
        global vec2
        self.xs = [x1,x2,x3,x4]
        self.ys = [displayHeight - y1,displayHeight - y2,displayHeight - y3,displayHeight - y4]
        self.active = True
        self.lines = []
        for i in range(4):
            self.lines.append(Line(self.xs[i%4],self.ys[i%4],self.xs[(i+1)%4],self.ys[(i+1)%4]))
            self.lines[i].setLineThinkness(1)
            self.lines[i].setColor([0, 0, 255])

        self.center = vec2((self.xs[0] + self.xs[2]) / 2, (self.ys[0] + self.ys[2]) / 2)
        print(self.center)

    def draw(self):
        if self.active:
            for i in self.lines:
                i.draw()

    #перевіряє на перетин з дроном

    def hitDrone(self, drone):
        if not self.active:
            return False

        global vec2
        rightVector = vec2(drone.direction)
        upVector = vec2(drone.direction).rotate(-90)
        droneCorners = drone.getCorners(rightVector, upVector)
        for i in range(4):
            j = i + 1
            j = j % 4
            if linesCollided(self.center[0]-20, self.center[1]-20, self.center[0]+20, self.center[1]+20, droneCorners[i].x, droneCorners[i].y, droneCorners[j].x, droneCorners[j].y):
                return True
        return False


class Drone:

    def __init__(self, walls, rewardGates,destination):
        global vec2
        self.nbVect = 16
        self.angles = np.linspace(-180, 180, self.nbVect)
        self.x = 670
        self.y = 580
        self.vel = 0
        self.direction = vec2(0.5, -1)
        self.direction = self.direction.rotate(180 / 12)
        self.acc = 0
        self.width = 30
        self.height = 30
        self.turningRate = 5.0 / self.width
        self.friction = 0.98
        self.maxSpeed = self.width / 4.0
        self.maxReverseSpeed = self.maxSpeed / 16.0
        self.accelerationSpeed = self.width / 160.0
        self.dead = False
        self.done = False
        self.lineCollisionPoints = []
        self.collisionLineDistances = []
        self.vectorLength = 600
        self.dronePic = pyglet.image.load('Drone.png')
        self.droneSprite = pyglet.sprite.Sprite(self.dronePic, x=self.x, y=self.y)
        self.droneSprite.update(rotation=0, scale_x=self.width / self.droneSprite.width,scale_y=self.height / self.droneSprite.height)
        self.turningLeft = False
        self.turningRight = False
        self.accelerating = False
        self.reversing = False
        self.walls = walls
        self.rewardGates = rewardGates
        self.destination = destination
        self.rewardNo = 0

        self.directionToRewardGate = self.rewardGates[self.rewardNo].center - vec2(self.x, self.y)

        self.reward = 0


    def reset(self):
        global vec2
        self.x = 650
        self.y = 580
        self.vel = 0
        self.direction = vec2(0.5, -1)
        self.direction = self.direction.rotate(180 / 12)
        self.acc = 0
        self.dead = False
        self.done = False
        self.lineCollisionPoints = []
        self.collisionLineDistances = []
        self.turningLeft = False
        self.turningRight = False
        self.accelerating = False
        self.reversing = False
        self.rewardNo = 0
        self.reward = 0
        for g in self.rewardGates:
            g.active = True

    def show(self):
        upVector = self.direction.rotate(90)
        drawX = self.direction.x * self.width / 2 + upVector.x * self.height / 2
        drawY = self.direction.y * self.width / 2 + upVector.y * self.height / 2
        self.droneSprite.update(x=self.x - drawX, y=self.y - drawY, rotation=-get_angle(self.direction))
        self.droneSprite.draw()
        # self.showCollisionVectors()


    #повертає вектор того, де знаходиться точка на дроні після обертання

    def getPositionOnDroneRelativeToCenter(self, right, up):
        global vec2
        w = self.width
        h = self.height
        rightVector = vec2(self.direction)
        rightVector.normalize()
        upVector = self.direction.rotate(90)
        upVector.normalize()

        return vec2(self.x, self.y) + ((rightVector * right) + (upVector * up))

    def updateWithAction(self, actionNo):
        self.turningLeft = False
        self.turningRight = False
        self.accelerating = False
        self.reversing = False

        if actionNo == 0:
            self.turningLeft = True
        elif actionNo == 1:
            self.turningRight = True
        elif actionNo == 2:
            self.accelerating = True
        elif actionNo == 3:
            self.reversing = True
        elif actionNo == 4:
            self.accelerating = True
            self.turningLeft = True
        elif actionNo == 5:
            self.accelerating = True
            self.turningRight = True
        elif actionNo == 6:
            self.reversing = True
            self.turningLeft = True
        elif actionNo == 7:
            self.reversing = True
            self.turningRight = True
        elif actionNo == 8:
            pass
        totalReward = 0
        for i in range(1):
            if not self.dead:
                self.lifespan += 1
                self.move()
                self.updateControls()
                if self.hitAWall():
                    self.dead = True
                elif self.hitADestination():
                    self.dead = True
                    self.done = True
                self.checkRewardGates()
                totalReward += self.reward

        self.setVisionVectors()
        self.reward = totalReward
        # self.update()


    def update(self):
        if not self.dead:
            self.updateControls()
            self.move()

            if self.hitAWall():
                self.dead = True
            elif self.hitADestination():
                self.dead = True
                self.done = True
            self.checkRewardGates()
            self.checkDestinationZone()
            self.setVisionVectors()


            if self.hitADestination():
                self.dead = True
                self.done = True
                # return

    #перевіряє на зіткнення з воротами з винагородою

    def checkRewardGates(self):
        global vec2
        self.reward = -1
        for i in range(len(self.rewardGates)):
            if self.rewardGates[self.rewardNo].hitDrone(self):
                self.rewardGates[self.rewardNo].active = False
                self.rewardNo += 1
                self.reward = 10
                if self.rewardNo == len(self.rewardGates):
                    self.rewardNo = 0
                    for g in self.rewardGates:
                        g.active = True
            self.directionToRewardGate = self.rewardGates[self.rewardNo].center - vec2(self.x, self.y)

    #перевіряє на зіткнення з кінцевою зоною

    def checkDestinationZone(self):
        global vec2
        self.reward = -1
        if self.rewardNo == self.rewardGates:
            for i in range(len(self.destination)):
                if self.destination[i].hitDrone(self):
                    self.reward = 100

    #переміщеє дрон відповідно до виконаної дії

    def move(self):
        global vec2
        self.vel += self.acc
        self.vel *= self.friction
        self.constrainVel()

        addVector = vec2(0, 0)
        addVector.x += self.vel * self.direction.x
        addVector.y += self.vel * self.direction.y

        if addVector.length() != 0:
            addVector.normalize()

        addVector.x * abs(self.vel)
        addVector.y * abs(self.vel)

        self.x += addVector.x
        self.y += addVector.y

    #зберігає швидкість в межах максимальної та мінімальної

    def constrainVel(self):
        if self.maxSpeed < self.vel:
            self.vel = self.maxSpeed
        elif self.vel < self.maxReverseSpeed:
            self.vel = self.maxReverseSpeed

    #оновлює керування дроном відповідно до встановлених дій

    def updateControls(self):
        multiplier = 1
        if abs(self.vel) < 5:
            multiplier = abs(self.vel) / 5
        if self.vel < 0:
            multiplier *= -1

        if self.turningLeft:
            self.direction = self.direction.rotate(radiansToAngle(self.turningRate) * multiplier)

        elif self.turningRight:
            self.direction = self.direction.rotate(-radiansToAngle(self.turningRate) * multiplier)
        self.acc = 0
        if self.accelerating:
            if self.vel < 0:
                self.acc = 3 * self.accelerationSpeed
            else:
                self.acc = self.accelerationSpeed
        elif self.reversing:
            if self.vel > 0:
                self.acc = -2 * self.accelerationSpeed
            else:
                self.acc = 0
                self.vel = 0

    #перевіряє на зіткнення з кожною стіною, та вертає True, якщо така є

    def hitAWall(self):
        for wall in self.walls:
            if wall.hitDrone(self):
                return True

        return False

    #перевіряє на зіткнення з кінцевою зоною

    def hitADestination(self):

        if self.destination[0].hitDrone(self):
            return True

        return False

    #повертає точку перетину вектора(x1,y1,x2,y2) зі стінами
    #якщо точок перетину зі стінами декілька, то вертається найближча точка до дрону

    def getCollisionPointOfClosestWall(self, x1, y1, x2, y2):
        global vec2
        minDist = 2 * displayWidth
        closestCollisionPoint = vec2(0, 0)
        for wall in self.walls:
            collisionPoint = getCollisionPoint(x1, y1, x2, y2, wall.x1, wall.y1, wall.x2, wall.y2)
            if collisionPoint is None:
                continue
            if dist(x1, y1, collisionPoint.x, collisionPoint.y) < minDist:
                minDist = dist(x1, y1, collisionPoint.x, collisionPoint.y)
                closestCollisionPoint = vec2(collisionPoint)
        return closestCollisionPoint

    #шляхом створення ліній у багатьох напрямках від дрона та
    #отримання найближчої точки зіткнення цієї лінії,
    #ми створюємо «вектори бачення», які дозволять дрону «бачити», тобто лідар

    def getState(self):
        self.setVisionVectors()
        normalizedVisionVectors = [1 - (max(1.0, line) / self.vectorLength) for line in self.collisionLineDistances]

        normalizedForwardVelocity = max(0, (self.vel - self.maxReverseSpeed) / (self.maxSpeed - self.maxReverseSpeed))
        normalizedPosDrift = 0
        normalizedNegDrift = 0


        normalizedAngleOfNextGate = (get_angle(self.direction) - get_angle(self.directionToRewardGate)) % 360
        if normalizedAngleOfNextGate > 180:
            normalizedAngleOfNextGate = -1 * (360 - normalizedAngleOfNextGate)

        normalizedAngleOfNextGate /= 180

        normalizedState = [*normalizedVisionVectors, normalizedForwardVelocity,
                           normalizedPosDrift, normalizedNegDrift, normalizedAngleOfNextGate]
        return np.array(normalizedState)

    def setVisionVectors(self):
        self.collisionLineDistances = []
        self.lineCollisionPoints = []
        for i in self.angles:
            self.setVisionVector(0, 0, i)

    #обчислює та зберігає відстань до найближчої стіни за заданим вектором

    def setVisionVector(self, startX, startY, angle):
        collisionVectorDirection = self.direction.rotate(angle)
        collisionVectorDirection = collisionVectorDirection.normalize() * self.vectorLength
        startingPoint = self.getPositionOnDroneRelativeToCenter(startX, startY)
        collisionPoint = self.getCollisionPointOfClosestWall(startingPoint.x, startingPoint.y,
                                                             startingPoint.x + collisionVectorDirection.x,
                                                             startingPoint.y + collisionVectorDirection.y)
        if collisionPoint.x == 0 and collisionPoint.y == 0:
            self.collisionLineDistances.append(self.vectorLength)
        else:
            self.collisionLineDistances.append(
                dist(startingPoint.x, startingPoint.y, collisionPoint.x, collisionPoint.y))
        self.lineCollisionPoints.append(collisionPoint)

    #візуалізує точки зіткнення векторів зі стінами

    def showCollisionVectors(self):
        global drawer
        for point in self.lineCollisionPoints:
            drawer.setColor([255, 0, 0])
            drawer.circle(point.x, point.y, 5)

    def getCorners(self, rightVector, upVector):
        droneCorners = []
        cornerMultipliers = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        dronePos = vec2(self.x, self.y)
        for i in range(4):
            droneCorners.append(dronePos + (rightVector * self.width / 2 * cornerMultipliers[i][0]) +
                                (upVector * self.height / 2 * cornerMultipliers[i][1]))
        return droneCorners