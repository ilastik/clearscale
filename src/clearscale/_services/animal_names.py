import random

CHARACTER_ADJECTIVES = """
agile amiable ardent aware brave calm canny chatty cheerful chipper civil clever comic cool coy crafty cuddly
dapper dainty daring deft eager earnest fabled fancy feisty fierce fiery firm frank frisky gallant gentle giddy
glad glorious goodly gracious grand gritty happy hearty honest humble hyper jazzy jovial just keen kindly lively
lucky mellow merry mighty mild modest nimble noble peppy perky plucky polite proud quick quirky ready regal
serene sassy saucy sharp shrewd sincere sleepy smiley snappy spicy spry sturdy tidy trusty upbeat urbane vivid
warm wily witty young zany zealous zesty zippy blithe bouncy dandy dreamy frolic frothy loony
""".split()

APPEARANCE_ADJECTIVES = """
airy ashen beady bland blotchy brassy bright brindled broad brushy bumpy chalky chunky clean cloudy coarse creamy
crisp crooked curved dappled dark dashed dense dusky feathery filmy flashy fluffy foamy frosty fuzzy gilded
glassy glossy golden grainy gray green grubby grimy grooved hairy hazy hefty inky jagged lacy leafy lean lemony
limpid lunar matte milky misty mottled muddy narrow neon nubby oily pale pebbled pearly petite plaid plush puffy
quilted ragged rainy rough round ruddy rustic sandy scaly sepia shaggy shiny sleek slim smoky spiky spotted striped
sunlit sweaty tawny textured tiny toasty tufty velvet velvety waxy whitish wispy woolly
""".split()

CREATURE_NOUNS = """
ant ape auk bat bee boa boar bud bug calf cat clam colt cow crow cub dace deer dog dove drake eel elk fawn ferret
fish flea fox frog gecko gnat goat goose gull hare hawk ibis imp jay koi lamb lark lemur lion lizard mink mole
monkey moth mouse mule newt otter owl panda parrot peacock piglet pixie pony puma quail ram raven seal shark shrew
skink slug snake sparrow spider sprite squid stoat swan tiger toad trout turtle viper vole weasel whale wolf wren
yeti yak zebra fern moss lichen coral kelp algae yeast spore bloom orchid tulip
""".split()

WILDCARD_ADJECTIVES = """
absurd acidic adroit aglow amped antsy apish astral atomic awry beamy beefy beepy bendy bimsy blobby blinky bloopy
boffo boingy bonkers boozy breech briny bubbly buggy bulgy buzzy chancy cheeky chirpy clanky cloaky clumsy comical
cosmic cranky creepy crazed cryptic daffy dizzy drifty droopy eerie edgy elvish fizzy flakey flimsy flouncy funky
gawky glitchy gnarly goofy grumpy gushy hatchy hinky hokey hoppy huffy jumpy jingly kooky lanky loopy louche mucky
murky nerdy noisy oddly plonky punky rambly ribby rowdy snarky spooky squiffy trippy wacky weirdy wonky yowzy
zappy bazzy blusty brisky clacky dorky drippy gloopy groovy grungy jumby kicky mangy nippy
""".split()


def generate_random_animal_name(seed: int | str | None = None) -> str:
    """
    Returns names consisting of a character adjective, appearance adjective, and animal noun.
    Each adjective has a 1% chance of being a quirky wildcard adjective instead.

    Examples:
      brave-shiny-otter
      witty-fuzzy-moss
      glitchy-sleek-drake
    """
    rng = random.Random(seed) if seed is not None else random.SystemRandom()

    if rng.random() < 0.01:
        adj1 = rng.choice(WILDCARD_ADJECTIVES)
    else:
        adj1 = rng.choice(CHARACTER_ADJECTIVES)

    if rng.random() < 0.01:
        adj2 = rng.choice(WILDCARD_ADJECTIVES)
    else:
        adj2 = rng.choice(APPEARANCE_ADJECTIVES)

    creature = rng.choice(CREATURE_NOUNS)

    return f"{adj1}-{adj2}-{creature}"
