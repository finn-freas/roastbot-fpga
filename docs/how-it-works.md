# How it works

[← back to the README](../README.md)

The board is a Sipeed Tang Nano 20K. It's got a small Gowin FPGA (about 20K LUTs, 48
multipliers, 46 block RAMs) and 8 MB of SDRAM. That's really tiny for a language model, so
pretty much everything here comes down to two things: fitting the model in 8 MB, and getting
it out of the SDRAM fast enough.

## The model

It's a small Llama-style transformer, shrunk down until it fit on the board:

| | |
|-|-|
| Layers | 16 |
| Width | 256 |
| Attention | 8 query heads sharing 2 key/value heads |
| Feed-forward | SwiGLU, 704 wide |
| Vocabulary | 3,072 word pieces (with room for emoji) |
| Context | 512 word pieces |
| Weights | 4-bit |

There's no floating point on the FPGA, so everything is integer math. The tricky parts
(softmax, normalizing, the sigmoid) use small lookup tables instead. `src/board.py` does the
exact same math in Python, and the board has to match it bit for bit.

## How I trained it

1. **Teach it to talk.** First I trained a base model from scratch on a lot of everyday
   conversation (the SODA dataset plus Reddit reply chains). This just teaches it English and
   how chats go back and forth.
2. **Get a bigger model to write roasts.** A model this small can't learn roasting from a few
   examples, so a bigger 4B model (running locally) wrote around 10,000 short chats where
   someone talks to the bot and it roasts them back. I showed it funny TikTok comments and a
   bunch of comebacks so it got the tone right. No slurs, nothing about groups of people, no
   suicide jokes.
3. **Throw out the bad ones.** The big model went back over every chat and scored whether the
   bot's lines actually made sense. The worst ones got dropped, along with copies of the
   examples and "your mom" lines that didn't come right after an insult.
4. **Fine-tune.** Then the small model got trained on the roast chats (with a little normal
   chat mixed in so it doesn't forget how to talk). It also learns from the big model's top
   guesses at each word, not just the one word that got written, which helps a lot at this
   size.
5. **Pack it.** Last, the model gets squished to 4-bit and packed into one 8 MB image for the
   board's memory.

`scripts/train_roastbot.sh` runs the whole thing (you'll need a GPU with 8 GB). The base model,
the roast chats, and the finished model are all in the repo, so you can start from any step,
like fine-tuning it on your own chats.

## How it picks a reply

1. **It reads your message.** Your message gets split into word pieces and sent to the board,
   which runs them through all 16 layers and remembers them.
2. **It writes a few replies.** The board writes up to 6 different replies, picking each word
   at random from the likely ones. It also scores how natural each reply is, with a little
   bonus for less common words so "lol ok" doesn't win every time.
3. **It throws out the bad ones.** Replies that loop, repeat themselves, or break the rules
   (slurs, digs at groups of people, "your mom" lines at the wrong time) get dropped.
4. **It checks that it's actually answering you.** Small models love safe replies that fit
   after anything ("fuck off, you're a loser"). So the 3 best replies get scored again as if
   you'd said nothing, and the one that depends most on what *you* said wins. In blind tests
   this picked the better reply 34 to 13.
5. **It remembers.** The winning reply goes back into the chat memory, so it remembers what it
   said.

**"Your mom" lines** only show up right after you insult it, and never too close together.
When one's allowed, the bot starts one candidate with "ur mom" or "your mom" and finishes it
itself, and it only wins if it's actually as good as the normal replies.

## How the hardware works

The model is way too big for the FPGA's own memory, so the whole 8 MB gets streamed out of
the SDRAM for every single word piece. Basically every design choice is about using that
stream well.

The model is packed like a little program: the weights, the word table, the lookup tables,
room for the chat memory, and a list of steps (multiply this, normalize that, run attention
for layer 5...). The engine on the FPGA reads that list and runs it once per word piece.

| Block | What it does |
|-|-|
| `engine` | runs the list of steps and drives everything else |
| `mv_unit` | the matrix math: 16 4-bit weights per clock, straight out of the SDRAM |
| `vec_unit` | the small stuff: normalizing, positions, adds, the SwiGLU gate |
| `attn_unit` | attention over the chat memory |
| `sampler` | picks the next word and scores it |
| `sdram_ctrl` | the SDRAM controller, at 108 MHz |
| `host_link`, `uart` | talking to the laptop over USB |

### The matrix unit

Every weight matrix is stored row by row. Each row starts with a 16-bit multiplier and a shift
for the whole row, then one 8-bit scale for every group of 32 weights, then the weights
themselves, packed 16 4-bit weights to a 64-bit word. For every word that comes out of the
SDRAM:

1. 16 multipliers multiply the 16 weights by their 16-bit inputs, and the results get added up
2. every 32 inputs, that partial sum gets multiplied by the group's scale and added to the
   row's total
3. at the end of the row, the total gets multiplied by the row's multiplier, shifted, rounded,
   and clamped back down to 16 bits

The inputs sit in a small RAM split into 16 lanes, so all 16 come out in the same clock. The
word table goes through the same unit with 8-bit weights, 8 per word.

### Running the SDRAM twice as fast

The SDRAM bus is only 32 bits wide, but the matrix unit wants 64 bits every clock, so at 54 MHz
it was sitting around half the time. So I run the SDRAM controller at 108 MHz, which is
exactly one 64-bit word per 54 MHz clock. The PLL makes both clocks from the board's 27 MHz
oscillator. That took a reply from 7.6 s to 4.8 s, with the exact same replies.

**Crossing between the two clocks.** Requests and read data go between the clocks through an
async FIFO in block RAM. The FIFO's read and write pointers cross over in Gray code, where only
one bit changes at a time, so if a pointer gets caught halfway through changing it's still only
off by one at most. Control signals cross with a toggle handshake.

**SDRAM timing.** Above 100 MHz the SDRAM needs a CAS latency of 3 instead of 2. Its internal
access takes about 18–20 ns, and 2 clocks at 108 MHz is only 18.5 ns. All the other SDRAM
timings (opening a row, precharge, refresh) get counted again for the faster clock.

**Catching the read data.** At 108 MHz one clock is only 9.3 ns. The data comes back almost a
whole clock late (the clock has to go out through a pin, the SDRAM has to do its access, and the
data has to come back in through the pins), and each word is only steady on the wires for part
of the clock. The data gets caught in registers right at the pins, and the PLL can shift the
phase of the clock going to the SDRAM, which slides that window earlier or later.

- At power-up, the controller writes a test pattern and reads it back with every read latency,
  then picks the middle of the ones that work.
- For every new build, `scripts/sdram_diag.py` sweeps the phase with 130 million words at each
  setting, and the build uses the middle of the phases with zero errors (1 to 5 for the current
  one, so it uses 3).
- `scripts/repack_phase.sh` changes the phase in a finished build without redoing place and
  route.

### Other things that made it faster

- **Letting the board write on its own.** The board's USB chip only passes data to the laptop
  every few milliseconds, so waiting on the laptop between every word was almost half the
  reply time. Now the laptop asks for a whole reply at once and the board just keeps going.
  Same replies, about 5 s instead of 6.
- **Fitting it in.** The design uses 76% of the FPGA, so most placement tries fail and each
  build tries a few seeds.

## Results

| | |
|-|-|
| FPGA usage | 76% of the LUTs, 43 of 46 block RAMs, 34 of 48 multipliers |
| Memory | 8,038 KiB of the 8,188 KiB SDRAM |
| Speed | about 21.5 ms per word piece, about 5 s per reply |
| Accuracy | 14 of 14 test replies identical to the CPU, and 14 of 14 insults |

### Why not bigger?

16 layers is the max for this board. Each extra layer needs another 448 KiB and there's only
150 KiB left.

### Would a TPU-style chip help?

Nope. Writing a word means reading every weight once, and the multipliers already keep up with
the SDRAM; they're waiting on memory, not math. A TPU only helps if you can reuse each weight
for a bunch of words at once, and the bot can't write its candidate replies side by side
without changing what it says. Plus the FPGA is basically full anyway.

## Testing

`src/board.py` is the reference: it does the exact same integer math as the board, down to the
rounding and the random numbers, so the same seed gives the same reply on the CPU and on the
board.

- `scripts/sim.sh link|mv|engine` simulates the hardware and checks it against `src/board.py`.
- `scripts/board_check.py --best` sends 14 messages to the board and the CPU and makes sure
  every reply matches. Every hardware change has to pass it.
- `./chat.sh` without `--board` runs the reference model, so you can try a model before it
  goes on the board.

The first time you run `chat.sh --board`, it uploads the model to the board (about a minute).
After that it's already there until you unplug it.
