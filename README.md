# roastbot-fpga

A chatbot that roasts you, running entirely on a $50 FPGA board (Sipeed Tang Nano 20K). It was first built as a chatbot trained on one person's texts, and then reduced to a bot that only roasts the person talking to it. The hardware is written to hold the whole model on the FPGA, so the board works on its own.

The replies are deliberately rude.

```
you> fuck you
roast bot> you're crying because a tiny board laughed at your ass
you> ur just a bot
roast bot> so you lost to a tiny little piece of plastic and still can't get a joke? fuckin pathetic 😤
```

It's a 16-layer, 12-million-parameter language model, and the FPGA does all of it.
The laptop just sends your message over USB and prints the answer (but you don't need a laptop!)

- **12,067,072 parameters** (16 layers, 4-bit weights)
- a reply takes about **5 seconds** (the board writes each candidate reply word after word on its own, instead of waiting for the laptop between words)
- the model fills the board's whole 8 MB of memory
- the board's replies match a Python model of the hardware bit for bit
- built with the open-source tools (Yosys, nextpnr, Verilator, cocotb), no vendor IDE

**[How it works →](docs/how-it-works.md)** (the model, how it was trained, the hardware, and results)

## Running it

The trained model and a tested bitstream are in the repo, so you don't need to train or build
anything:

```bash
pip install numpy torch tokenizers
./chat.sh                                                    # chat with it on your computer
openFPGALoader -b tangnano20k -f bitstream/roastbot.fs       # or write the bitstream to the board's flash
./chat.sh --board                                            # and chat with it on the board
```

`./chat.sh` runs the exact same integer math as the board, so you get the same replies
without one. Every chat starts from seed 1, so the same messages always get the same
replies. `./chat.sh --seed random` picks a new seed and prints it, so you can replay a chat later
with `--seed <number>`. To build the bitstream yourself, run `scripts/build_board.sh`. Not every
placement seed places, so try a few (`SEED=6 scripts/build_board.sh`), and check the SDRAM
timing on the board with `scripts/sdram_diag.py` before you trust it.

## Chat commands

Lines that start with `/` are commands, not messages:

| Command | |
|-|-|
| `/reset` | clear the conversation |
| `/temp 0.6`, `/temp auto` | sampling temperature, which must be above 0 (`auto` goes back to the default) |
| `/cut 4`, `/cut 0` | only words within this distance of the best one can be picked, so lower is safer and higher is wilder; 0 or more |
| `/emoji 0.5` | how many emoji, where 0.5 is the usual amount, 1 is more and 0 is none |
| `/best 6`, `/best auto` | how many candidate replies to write per message (1 turns it off); the one that reads most like the bot is shown |
| `/seed 7`, `/seed random` | restart the random numbers from a seed |
| `/info` | the layer count, tokens used, current settings and SDRAM size |
| `/quit` | leave (so do `/q` and `/exit`) |

A bad value prints a message and leaves the setting as it was.

## Checking the board

`link.py`, `chat.py --board` and the board scripts open the Tang Nano 20K's second USB serial port at 1,000,000 baud. The port is the `/dev/serial/by-id/*SIPEED*if01*` device when one exists and `/dev/ttyUSB1` otherwise. If the port cannot be opened, the tools print the reason and exit.

```bash
python src/link.py ping                          # print the link version and SDRAM status
python src/pack.py models/pr_board.pkl           # write models/pr_board.sdram.bin
python src/link.py load models/pr_board.pkl      # upload that image and verify its checksum
python scripts/board_check.py [model] [--best] [--argue]   # compare board replies with the CPU
python scripts/sdram_diag.py [rounds [words]]    # SDRAM read/write test (default 20 rounds, 65536 words)
```

`chat.py --board` uploads the model itself when the board does not hold it. `FB_MODELS` points the chat tools at another models directory.

## What's in here

| Folder | What it has |
|-|-|
| `src/` | the Python side. `board.py` is the reference model: the exact integer math the hardware does, used to check the board bit for bit. `pack.py` packs a model into the 8 MB SDRAM image. `engine.py` runs that image the way the hardware engine does, op by op, and counts the clock cycles. `link.py` talks to the board over USB. `chat.py` is the chat program (picking replies, "your mom" rules, the block list). `tok.py` and `corpus.py` build the tokenizer and the everyday-chat data, `train.py` trains the base model, and `finetune.py` teaches it to roast |
| `rtl/` | the hardware in SystemVerilog: the engine and its units (matrix, vector, attention, sampler, SDRAM read stream) |
| `rtl/board/` | the parts specific to the Tang Nano 20K: the top level, SDRAM controller, clock crossing, PLLs, UART, pin constraints and clock constraints |
| `rtl/sim/` | simulation tops and an SDRAM model, for the tests in `tb/` |
| `tb/` | cocotb testbenches that run the hardware in Verilator and check it against `src/board.py` (run with `scripts/sim.sh link`, `mv` or `engine`) |
| `tests/` | plain Python unit tests (pytest) that need no weights, datasets, GPU or board: the integer math in `board.py` and `engine.py` and their agreement on a tiny random model, the `pack.py` image layout, the `link.py` framing and argument checks, the tokenizer, the `corpus.py` builders, and the argument handling of the chat and diagnostic scripts. Run them with `python -m pytest -q tests` (needs `numpy torch tokenizers pyarrow pytest`) |
| `scripts/` | everything else you run: building the bitstream (`build_board.sh`), checking the board (`board_check.py`, `sdram_diag.py`), the whole training recipe (`train_roastbot.sh`) and its steps (writing roast chats, filtering them, emoji, distillation targets), and evaluating replies |
| `models/` | the trained models: `pr_board.pkl.xz` is the quantized model the board runs (unpacked the first time you chat), `pr.pt` is the same model before quantizing, and `pre_b16.pt` is the base model before it learned to roast |
| `data/people/pr/` | the ~8,100 roast chats it was fine-tuned on, all written by the writer model |
| `data/tok/` | the tokenizer |
| `data/style/` | the TikTok comments and comebacks shown to the writer model for the tone |
| `bitstream/` | `roastbot.fs`, the tested bitstream (link version 4, placement seed 3, SDRAM clock phase 3; 14 of 14 replies identical to the CPU, plus 14 of 14 insults) |
| `docs/` | [how it works](docs/how-it-works.md): the model, the training, the hardware, and the results |

`chat.sh` at the top just runs `src/chat.py`.

To train your own from scratch, run `scripts/train_roastbot.sh` (needs a GPU with 8 GB).
