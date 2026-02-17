import pygame
import random
from game_brain1 import Brain

WIDTH, HEIGHT = 600, 400
BIRD_X = 50
PIPE_SPEED = 5
GENERATION = 1

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 20)

class Bird:
    def __init__(self, parent_brain=None):
        self.y = HEIGHT // 2
        self.velocity = 0
        self.brain = Brain() # every bird get a random brain
        self.alive = True
        self.score = 0
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))

        if parent_brain:
            self.brain.weights = parent_brain.weights[:] 
            self.brain.bias = parent_brain.bias
            self.brain.mutate()
        else:
            pass

    def update(self, pipe_x, pipe_height):
        if not self.alive: return

        # gravity
        self.velocity += 0.5
        self.y += self.velocity
        self.score += 1 # surviving longer = high score

        # think
        inputs = [
            self.y / HEIGHT,
            pipe_x / WIDTH,
            pipe_height / HEIGHT
        ]

        decision = self.brain.think(inputs)

        if decision > 0:
            self.velocity = -8

        #death : hit ground or ceiling
        if self.y > HEIGHT or self.y < 0:
            self.alive = False

        # hit pipe (collision)
        if pipe_x < BIRD_X + 20 < pipe_x + 50:
            if self.y < pipe_height or self.y > pipe_height + 100: #100 is gap size
                self.alive = False

def run_generation(best_brain):
    global GENERATION
    birds = []
    for _ in range(30):
        # Pass the best brain from the last round to the new birds
        birds.append(Bird(parent_brain=best_brain))
    
    #pipe setup
    pipe_X = WIDTH
    pipe_height = random.randint(50, 250)

    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        screen.fill((0,0,0))
        pipe_X -= PIPE_SPEED
        if pipe_X < -50:
            pipe_X = WIDTH
            pipe_height = random.randint(50, 250)

        #update birds
        alive_count = 0
        for bird in birds:
            if bird.alive:
                bird.update(pipe_X, pipe_height)
                alive_count += 1
                pygame.draw.circle(screen, bird.color, (BIRD_X, int(bird.y)), 10)
        
        #draw pipe
        pygame.draw.rect(screen, (0, 255, 0), (pipe_X, 0, 50, pipe_height))
        pygame.draw.rect(screen, (0, 255, 0), (pipe_X, pipe_height + 100, 50, HEIGHT))

        # --- UI ---
        text = font.render(f"Gen: {GENERATION} | Alive: {alive_count}", True, (255, 255, 255))
        screen.blit(text, (10, 10))
        pygame.display.flip()
        clock.tick(30) # 30 FPS

        # If all birds are dead, evolve!
        if alive_count == 0:
            running = False
            best_bird = max(birds, key=lambda b: b.score)
            print(f"Gen {GENERATION} Best Score: {best_bird.score}")
            GENERATION += 1
            
            return best_bird.brain

# --- MAIN LOOP ---
current_best_brain = None

while True:
    # Run the game, and capture the winner's brain
    current_best_brain = run_generation(current_best_brain)