import random
import time
import pygame.midi

# --- CONFIGURATION ---
SONG_LENGTH = 50
POPULATION_SIZE = 100
MAX_GENERATIONS = 1000

# --- THE MENU OF OPTIONS ---
# The AI can pick a Single Note OR a Full Chord
OPTIONS = [
    # Single Notes (Melody)
    [60], [62], [64], [65], [67], [69], [71],
    # Full Chords (Harmony)
    [60, 64, 67], # C Major
    [65, 69, 72], # F Major
    [67, 71, 74], # G Major
    [69, 72, 76]  # A Minor
]

class Pianist:
    def __init__(self, length):
        # DNA is now a list of LISTS (e.g., [[60, 64, 67], [62]])
        self.genome = [random.choice(OPTIONS) for _ in range(length)]
        self.fitness = 0

    def calculate_fitness(self):
        score = 0
        
        # Rule 1: END ON C MAJOR (Resolution)
        # Ending on a full C Major chord is the best feeling in music (+20 points)
        last_event = self.genome[-1]
        if last_event == [60, 64, 67]:
            score += 20
        elif last_event == [60]:
            score += 5
        
        # Rule 2: VARIETY (Don't just play chords all the time)
        # We want a mix of chords and single notes.
        chord_count = sum(1 for x in self.genome if len(x) > 1)
        if 4 < chord_count < 10: # If 25-60% of the song is chords, that's good balance
            score += 10
        
        # Rule 3: SMOOTHNESS (Root note shouldn't jump too far)
        for i in range(len(self.genome) - 1):
            root_current = self.genome[i][0]   # Bottom note of current
            root_next = self.genome[i+1][0]    # Bottom note of next
            interval = abs(root_current - root_next)
            
            if interval > 7: 
                score -= 5 # Punishment for crazy jumps
            else:
                score += 2 # Reward for smooth movement

        self.fitness = score
        return score

    def mutate(self):
        if random.random() < 0.1:
            index = random.randint(0, len(self.genome) - 1)
            self.genome[index] = random.choice(OPTIONS)

    def crossover(self, partner):
        child = Pianist(len(self.genome))
        midpoint = random.randint(0, len(self.genome))
        child.genome = self.genome[:midpoint] + partner.genome[midpoint:]
        return child

def play_song(genome):
    try:
        pygame.midi.init()
        if pygame.midi.get_count() == 0:
            print("❌ No MIDI device found.")
            return

        player = pygame.midi.Output(0)
        player.set_instrument(0) # Acoustic Grand Piano
        
        print("\n🎹 Playing Polyphonic Masterpiece...")
        
        for event in genome:
            print(f"🎵 {event}")
            
            # 1. PRESS ALL KEYS IN THE CHORD
            for note in event:
                player.note_on(note, 100) # Volume 100
            
            # 2. WAIT (Let them ring together)
            time.sleep(0.6)
            
            # 3. RELEASE ALL KEYS
            for note in event:
                player.note_off(note, 100)
        
        del player
        pygame.midi.quit()
    except Exception as e:
        print(f"Audio Error: {e}")

def run_evolution():
    population = [Pianist(SONG_LENGTH) for _ in range(POPULATION_SIZE)]
    print(f"--- Evolution Started ({MAX_GENERATIONS} Gens) ---")

    for generation in range(1, MAX_GENERATIONS + 1):
        for p in population: p.calculate_fitness()
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        if generation % 20 == 0:
            best = population[0]
            print(f"Gen {generation} | Score: {best.fitness}")

        # Evolution Logic
        top_10 = population[:int(POPULATION_SIZE * 0.1)]
        new_population = []
        for _ in range(POPULATION_SIZE):
            parent_a = random.choice(top_10)
            parent_b = random.choice(top_10)
            child = parent_a.crossover(parent_b)
            child.mutate()
            new_population.append(child)
        population = new_population

    return population[0].genome

if __name__ == "__main__":
    winning_song = run_evolution()
    play_song(winning_song)