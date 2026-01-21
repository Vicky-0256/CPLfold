INF = 1000000007

NOTON = 5  # NUM_OF_TYPE_OF_NUCS
NOTOND = 25
NOTONT = 125

EXPLICIT_MAX_LEN = 4
SINGLE_MIN_LEN = 0
SINGLE_MAX_LEN = 30  # NOTE: *must* <= sizeof(char), otherwise modify State::TraceInfo accordingly

HAIRPIN_MAX_LEN = 30
BULGE_MAX_LEN = SINGLE_MAX_LEN
INTERNAL_MAX_LEN = SINGLE_MAX_LEN
SYMMETRIC_MAX_LEN = 15
ASYMMETRY_MAX_LEN = 28

# Vienna & CONTRAfold encoding
def get_acgu_num_v(x):
    return 1 if x == 'A' else 2 if x == 'C' else 3 if x == 'G' else 4 if x == 'U' else 0

def get_acgu_num_c(x):
    return 0 if x == 'A' else 1 if x == 'C' else 2 if x == 'G' else 3 if x == 'U' else 4

# lhuang: Vienna: 0N 1A 2C 3G 4U
# lhuang: CONTRA: 0A 1C 2G 3U 4N
# get_acgu_num = get_acgu_num_v 

# if lv else get_acgu_num_c

# if 'lv' in globals() else get_acgu_num_c
# print('globals')
# print(globals())

# Initialize the allowed pairs matrix
allowed_pairs = [[False] * NOTON for _ in range(NOTON)]

def initialize(lv):
    get_acgu_num = get_acgu_num_v if lv else get_acgu_num_c
          
    allowed_pairs[get_acgu_num('A')][get_acgu_num('U')] = True
    allowed_pairs[get_acgu_num('U')][get_acgu_num('A')] = True
    allowed_pairs[get_acgu_num('C')][get_acgu_num('G')] = True
    allowed_pairs[get_acgu_num('G')][get_acgu_num('C')] = True
    allowed_pairs[get_acgu_num('G')][get_acgu_num('U')] = True
    allowed_pairs[get_acgu_num('U')][get_acgu_num('G')] = True

# Call the initialize function to set up the allowed pairs
initialize(lv=False)
