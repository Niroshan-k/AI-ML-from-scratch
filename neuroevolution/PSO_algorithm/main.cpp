// Trick SDL3 into skipping its broken prefetch hack
#define __PRFCHWINTRIN_H

#include <SDL3/SDL.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <cstdint>

#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 720

float target_x = SCREEN_WIDTH / 2.0f;
float target_y = SCREEN_HEIGHT / 2.0f;

struct Particle {
    float x, y;
    float vx, vy;
    float best_x, best_y;
    float best_fitness;
    uint8_t r, g, b;

    Particle( float start_x, float start_y) {
        x = start_x;
        y = start_y;
        best_x = start_x;
        best_y = start_y;
        best_fitness = 999999.0f;
        vx = 1.0f;
        vy = 1.0f;

        r = 100 + rand() % 156;
        g = 100 + rand() % 156;
        b = 100 + rand() % 156;
    }
};

int main() {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        printf("couldn't initialize SDL: %s\n", SDL_GetError());
        return EXIT_FAILURE;
    }

    SDL_Window *window  = SDL_CreateWindow("Particle Swarn", SCREEN_WIDTH, SCREEN_HEIGHT, 0);
    if (!window) {
        printf("Failed to open %d x %d window: %s\n", SCREEN_WIDTH, SCREEN_HEIGHT, SDL_GetError());
        return EXIT_FAILURE;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, nullptr);

    bool running = true;
    SDL_Event event;

    std::vector<Particle> swarm;

    for (int i = 0; i < 50; i ++) {
        swarm.emplace_back(rand() % SCREEN_WIDTH, rand() % SCREEN_HEIGHT);
    }
    
    float global_best_x = swarm[0].x;
    float global_best_y = swarm[0].y;
    float global_best_fitness = 999999.0f;

    // loop
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) running = false;

            if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
                if (event.button.button == SDL_BUTTON_LEFT) {
                    target_x = event.button.x;
                    target_y = event.button.y;

                    global_best_fitness = 999999.0f;
                    for (auto& p : swarm) {
                        p.best_fitness = 999999.0f;

                        p.vx += ((rand() % 100) / 5.0f) - 30.0f; 
                        p.vy += ((rand() % 100) / 5.0f) - 30.0f;
                    }
                }
            }
        }

        // clear screen
        SDL_SetRenderDrawColor(renderer, 20, 20, 30, 255);
        SDL_RenderClear(renderer);

        for (auto& p : swarm) {

            float dist = (p.x - target_x) * (p.x - target_x) + (p.y - target_y) * (p.y - target_y);

            if ( dist < p.best_fitness) {
                p.best_fitness = dist;
                p.best_x = p.x;
                p.best_y = p.y;
            }

            if ( dist < global_best_fitness) {
                global_best_fitness = dist;
                global_best_x = p.x;
                global_best_y = p.y;
            }

            float r1 = (rand() % 100) / 100.0f;
            float r2 = (rand() % 100) / 100.0f;

            p.vx = (0.95f * p.vx) + (0.1f * r1 * (p.best_x - p.x)) + (0.05f * r2 * (global_best_x - p.x));
            p.vy = (0.95f * p.vy) + (0.1f * r1 * (p.best_y - p.y)) + (0.05f * r2 * (global_best_y - p.y));

            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0 || p.x > SCREEN_WIDTH)  p.vx *= -1.0f; // Reverse X direction
            if (p.y < 0 || p.y > SCREEN_HEIGHT) p.vy *= -1.0f; // Reverse Y direction

            float angle = atan2(p.vy, p.vx);
            float size = 12.0f;

            // Left back corner and Right back corner
            float p1_x = p.x - size * cos(angle - 0.5f);
            float p1_y = p.y - size * sin(angle - 0.5f);
            float p2_x = p.x - size * cos(angle + 0.5f);
            float p2_y = p.y - size * sin(angle + 0.5f);

            SDL_SetRenderDrawColor(renderer, p.r, p.g, p.b, 255);
            
            SDL_RenderLine(renderer, p.x, p.y, p1_x, p1_y);
            SDL_RenderLine(renderer, p.x, p.y, p2_x, p2_y);
            SDL_RenderLine(renderer, p2_x, p2_y, p1_x, p1_y);
            SDL_RenderLine(renderer, p.x, p.y, p.x - (p.vx * 4.0f), p.y - (p.vy * 4.0f));

            SDL_FRect p_rect = { p.x, p.y, 4, 4};
            SDL_RenderFillRect(renderer, &p_rect);

        }

        // target
        SDL_SetRenderDrawColor(renderer, 255, 50, 50, 255);
        SDL_FRect target_rect = { target_x - 5.0f, target_y - 5.0f, 10, 10};
        SDL_RenderFillRect(renderer, &target_rect);

        // update the screen
        SDL_RenderPresent(renderer);
        SDL_Delay(60);
    }

    SDL_DestroyWindow(window);
    SDL_DestroyRenderer(renderer);
    SDL_Quit();

    return 0;
}