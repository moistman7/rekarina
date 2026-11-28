// Pictures in the menu's description pane (ui/menu_pictures.cpp): drawn with the menu's own draw lists, font and colours,
// above the selected row's description, for the rows a picture explains better than a paragraph. Each takes the pane's
// draw list, the top left corner and the width it may use, and returns the height it drew (0: nothing).
//
//   picture pipeline  the 3D scene's size, what upscales it, the game picture, the window and how it is scaled into it
//   controller        the gamepad the game reads, live: buttons light up, the sticks' positions over their dead zones
//   keyboard          the keyboard and the mouse with the Switch button each bound key presses, the selected row's keys
//                     framed
//   performance       frame times of the last two seconds, CPU and GPU time per frame, video memory and RAM
//   system            graphics card, driver, video memory, processor, memory, display, the device profile and why
#pragma once
#include "imgui.h"
#include "ui/menu.h"

enum MenuPicture { PIC_NONE, PIC_PIPELINE, PIC_PAD, PIC_PERF, PIC_SYSTEM, PIC_KEYS };

#ifdef GAME_OOT
bool menu_fsr4_selected();
#endif

// The picture for the row of setting key (REKARINA_ name or row id, opt_desc_key), PIC_NONE for none.
MenuPicture menu_picture_for(const char *key);
// Draws it; key is the row described (the pipeline marks the stage that row sets), scale the menu's (1 at 1080 lines).
// The performance and system pictures read perfmon's numbers, which need its sampling on (menu.cpp keeps it on while
// one of them shows).
float menu_picture_draw(MenuPicture pic, const char *key, ImDrawList *dl, float x, float y, float w, float scale,
                        const MenuTarget *t);
