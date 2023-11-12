"""
The utilities.py file holds functions that are relevant for many different applications, either for setup
(e.g. getting Bravais vectors) or for defining useful functions (e.g. the Fermi function)
"""

import numpy as np
from numpy.polynomial.legendre import leggauss as GAULEG

from typing import Tuple

# Vectorized Fermi function
def fermi(E: np.ndarray, T: float, k_B: float) -> np.ndarray:
    if T < 1e-8:
        result = np.where(E > 0, 1.0, np.where(E == 0, 0.5, 0.0))
    else:
        term = E / (k_B * T)
        # Clip term values to avoid overflow/underflow
        clipped_term = np.clip(term, -500.0, 500.0)
        result = 1.0 / (np.exp(clipped_term) + 1.0)
    return result

# Function that returns a (2NCELLS+1)x(2NCELLS+1)x(2NCELLS+1) grid
def get_RPTS(a_1: np.ndarray, a_2: np.ndarray, a_3: np.ndarray, NCELLS: int) -> np.ndarray:
    
    rng = np.arange(-NCELLS, NCELLS + 1)
    i, j, k = np.meshgrid(rng, rng, rng, indexing='ij')
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    RPTS = (i[:, np.newaxis] * a_1 + j[:, np.newaxis] * a_2 + k[:, np.newaxis] * a_3)
    
    return RPTS

# Function that returns the k-mesh
def get_KPTS(a_1: np.ndarray, a_2: np.ndarray, a_3: np.ndarray, N_x: int, N_y: int, N_z: int) -> np.ndarray:
    
    # Calculate the volume of the unit cell
    volume = np.dot(a_1, np.cross(a_2, a_3))
    
    # Calculate the reciprocal space vectors
    b_1 = 2.0 * np.pi * np.cross(a_2, a_3) / volume
    b_2 = 2.0 * np.pi * np.cross(a_3, a_1) / volume
    b_3 = 2.0 * np.pi * np.cross(a_1, a_2) / volume
    
    # Calculate the total number of different wavevectors in reciprocal space
    Ntot = N_x * N_y * N_z
    
    # Initialize KPTS
    KPTS = np.zeros((Ntot, 3))
    
    # Generate ranges for c_1, c_2, c_3
    c_1_range = np.arange(1, N_x + 1) / N_x
    c_2_range = np.arange(1, N_y + 1) / N_y
    c_3_range = np.arange(1, N_z + 1) / N_z
    
    # Create a grid of c_1, c_2, c_3 values
    c_1, c_2, c_3 = np.meshgrid(c_1_range, c_2_range, c_3_range, indexing='ij')
    
    # Reshape the grid arrays to 1D arrays
    c_1, c_2, c_3 = c_1.ravel(), c_2.ravel(), c_3.ravel()
    
    # Calculate KPTS using broadcasting
    KPTS[:, 0] = b_1[0] * c_1 + b_2[0] * c_2 + b_3[0] * c_3
    KPTS[:, 1] = b_1[1] * c_1 + b_2[1] * c_2 + b_3[1] * c_3
    KPTS[:, 2] = b_1[2] * c_1 + b_2[2] * c_2 + b_3[2] * c_3
    
    return KPTS

# Function that counts the number of neighbours for every atom, given R_max
def get_nndists(RPTS: np.ndarray, TPTS: np.ndarray, R_max: float) -> np.ndarray:

    # Compute all ttprime differences (shape will be N_b x N_b x 3)
    ttprime = TPTS[:, np.newaxis, :] - TPTS[np.newaxis, :, :]
    
    # Compute the distance from all rpoints to each ttprime
    # Reshape RPTS for broadcasting (shape will be 1 x 1 x len(RPTS) x 3)
    RPTS_reshaped = RPTS.reshape((1, 1, -1, 3))
    # Calculate the distances (shape will be N_b x N_b x len(RPTS))
    distances = np.linalg.norm(RPTS_reshaped + ttprime[:, :, np.newaxis, :], axis=3)
    
    # Apply the distance criteria to count the number of neighbours
    # The distance criteria are applied within a 3D mask (shape will be N_b x N_b x len(RPTS))
    mask = (distances < R_max + 1e-5) & (distances > 1e-5)
    # Sum over the third axis to count the neighbors (shape will be N_b x N_b)
    num_neighbs = np.sum(mask, axis=2)
    
    # This is the symmetric matrix where the i,j element counts the number of j type neighbours that i type has
    # We simply need the total number of neighbours per atom, so we sum over the second axis
    num_neighbs = np.sum(num_neighbs, axis=1)
                    
    return num_neighbs

# Get the atom and distance connection vectors
def get_connections(max_neighb: int, RPTS: np.ndarray, TPTS: np.ndarray, R_max: float) -> Tuple[np.ndarray, np.ndarray]:

    N_b: int = TPTS.shape[0]
    # Arrays to hold the required results
    atom_IJ = np.zeros((N_b, max_neighb), dtype=int)
    Rvec_ij = np.zeros((N_b, max_neighb, 3))

    for i in range(N_b):
        added = 0
        for j in range(N_b):
            ttprime = TPTS[j] - TPTS[i]
            for rpoint in RPTS:
                vec = rpoint + ttprime
                dist = np.linalg.norm(vec)
                if (dist < R_max + 1e-5) and (dist > 1e-5):
                    # Get the type of atom j that is a neighbour of atom i
                    # excluding self-interactions
                    atom_IJ[i, added] = j
                    # Get the vector connecting atom j to atom i
                    Rvec_ij[i, added, :] = vec
                    added += 1

    return atom_IJ, Rvec_ij


# Function to get DoS using the energy eigenvalues and the electron eigenvectors as weights
# Surprisingly, vectorizing this in terms of energies results into a slower execution time
def get_dos(E_vecs: np.ndarray, E_vals: np.ndarray, intervals: int, N_b: int, Γ: float, N_k: int, vect: bool) -> Tuple[np.ndarray,np.ndarray]:

    # Get mesh
    Es = np.linspace(E_vals.min(),E_vals.max(),intervals)
    
    # Setup an array to hold the DoS values
    DOS = np.zeros((N_b,intervals))

    # Extract wavefunctions for electrons
    u_ups, u_downs = E_vecs[:, :N_b, :], E_vecs[:, N_b:2*N_b, :]
    u_up_squared = np.abs(u_ups)**2
    u_down_squared = np.abs(u_downs)**2

    # Check if user wants to do stuff vectorized - the downside is that the pc may not be
    # able to handle the operation memory-wise. If it can, vectorized *may* be faster on occasion
    if not vect:
        for idx, E in enumerate(Es):
            DOS[:,idx] = (Γ/np.pi)*(1.0/N_k)*((u_up_squared+u_down_squared)/(Γ**2 + (E_vals[:, np.newaxis, :]-E)**2)).sum(axis=(0,2))
    else:
        us = (u_up_squared+u_down_squared)[:, :, :, np.newaxis] # Shape (N_k, N_b, 2N_b, 1)
        E_vals_expanded = E_vals[:, np.newaxis, :, np.newaxis] # Shape: (N_k, 1, 2*N_b, 1)
        Es_expanded = Es[np.newaxis, np.newaxis, np.newaxis, :] # Shape: (1, 1, 1, intervals)

        # Vectorized calculation of DOS
        denominator = Γ**2 + (E_vals_expanded - Es_expanded)**2
        DOS = (Γ/np.pi) * (1.0/N_k) * (us / denominator).sum(axis=(0, 2))

    return Es, DOS

# This function allows us to re-shape the E_vecs and E_vals arrays so that we can go from
# [u↑, u↓, v↑, v↓]^T into -> [u1↑, u1↓, v1↑, v1↓, u2↑, u2↓, v2↑, v2↓, etc.]^T
def get_alt_vecs(E_vecs: np.ndarray) -> np.ndarray:

    # Get number of basis atoms
    N_b = E_vecs.shape[-1]//4
    # Find new indices
    indices = np.array([(i % 4) * N_b + i // 4 for i in range(4*N_b)])
    
    return E_vecs[:, indices, :]

# Function that returns an energy mesh for DoS calculations within the Green's function formalism
def get_dos_mesh(N: int, E_min: float, E_max: float, Γ: float) -> np.ndarray:

    Es = np.zeros((N), dtype=np.complex128)
    # Get evenly spaced real energies mesh
    real_mesh = np.linspace(E_min, E_max, N)
    # Get the imaginary displacement
    η = (0.0+1.0j)*Γ
    
    # Get Energies
    Es = real_mesh + η

    return Es

# This is a helper function that is used to retrieve an energy mesh for integrations below
# Needs further documentation, for reference see https://doi.org/10.1080/14786430802406256
def GAUFD(N: int) -> Tuple[np.ndarray, np.ndarray]:
    if N == 1:
        x = -49817229548128141768e-20
        w = 10000000000000031192e-19
    elif N == 2:
        x = [-78465071850839016234e-20, -20091536266094051757e-20]
        w = [50923235990870048433e-20, 49076764009130263488e-20]
    elif N == 3:
        x = [-88288518955458358024e-20, -48117621892777473749e-20, -88198184413497647625e-21]
        w = [28858444436509900908e-20, 45966895698954759346e-20, 25174659864535651667e-20]
    elif N == 4:
        x = [-92613063531202843773e-20, -64918327008663578157e-20,
             -28982568853420020298e-20, -24595209663255169680e-21]
        w = [18501429405165520392e-20, 34614391006511784214e-20,
             34152482191988153127e-20, 12731697396334854188e-20]
    elif N == 5:
        x = [-94875333872503463082e-20, -74805843506753178608e-20, -45504655263391074765e-20,
             -16657582360358973599e-20, 27402283545708211900e-21]
        w = [12939804504572789754e-20, 26102400189213290231e-20, 30851911091450589451e-20,
             24746815229701880449e-20, 53590689850617620359e-21]
    elif N == 6:
        x = [-96204950250095729781e-20, -80971428101130972258e-20, -57293627456482418171e-20,
             -30755197635518367504e-20, -82123839469384988331e-21, 83748358371240941581e-21]
        w = [96268650841705383829e-21, 20246201047059595265e-20, 26160719441051813381e-20,
             25781980698475975536e-20, 16683001513553609336e-20, 15012322156887800205e-21]
    elif N == 7:
        x = [-97053934379083423143e-20, -85045695849615413757e-20,  -65665104053460540522e-20, -42357896269371657364e-20,
             -19472732441816555564e-20, -19669621223691542539e-21, 15142830586888806919e-20]
        w = [74948008822570509041e-21, 16170863905729061704e-20, 22007120289205973485e-20, 23880411919774885893e-20,
             20952460047488907594e-20, 92465405554445737538e-21, 24780240009985858690e-22]
    elif N == 8:
        x = [-97630544447925725992e-20, -87873822716479965943e-20, -71736329217593360204e-20, -51463306578144813387e-20,
             -29967081434747298359e-20, -10763455942936048359e-20, 35963113675701677498e-21, 23003149140664609750e-20]
        w = [60394634019629989770e-21, 13252509350880929004e-20, 18643612522057003210e-20, 21413715867867937533e-20,
             21005092708864293339e-20, 16003068683842947897e-20, 36159126989806650464e-21, 26624765543536915040e-23]
    elif N == 9:
        x = [-98041275487012188695e-20, -89918326179154863440e-20, -76254129548477842110e-20, -58579104527384144901e-20,
             -38924212142470946276e-20, -19724340764961096691e-20, -40039281758884590381e-21, 97228170103579374416e-21,
             31678885353558278864e-20]
        w = [49992516372028853833e-21, 11099301824870447793e-20, 15971411690431220541e-20, 19037877203046567198e-20,
             19869087157813151863e-20, 17972334325952047726e-20, 10203571121909080322e-20, 84501828581921130722e-22,
             21467529556997868476e-24]
    elif N == 10:
        x = [-98345122025502045873e-20, -91446749996879318119e-20, -79700500547314513626e-20, -64189534981349313375e-20,
             -46376588343242516012e-20, -28030431525349494354e-20, -11327091328726333942e-20, 17437648086722052805e-21,
             16877498338102917782e-20, 40960465258252015313e-20]
        w = [42278597323639457484e-21, 94666349251635366832e-21, 13843777024241956101e-20, 16932936699837666261e-20,
             18398357022114735352e-20, 17939886390638648260e-20, 14468854182396060463e-20, 46026485095922891703e-21,
             11890402956686871419e-22, 14148408460516817666e-25]
    elif N == 11:
        x = [-98576901837451635280e-20, -92621727156102677473e-20, -82389243156123939088e-20, -68670708816882492198e-20,
             -52549052940365991088e-20, -35349156561982307316e-20, -18652071146560858606e-20, -45389164233559550280e-21,
             76984180593432347734e-21, 24899533750455431614e-20, 50711636785486806957e-20]
        w = [36383684790132198923e-21, 81985364434128201418e-21, 12133566247788805356e-20, 15122112006362489825e-20,
             16900090791849557413e-20, 17240157268363508589e-20, 15745585899461757802e-20, 97600157144810676257e-21,
             12496828256639735424e-21, 11876318920871395759e-23, 80046822403386311030e-27]
    elif N == 12:
        x = [-98758247347129831371e-20, -93546465146779806654e-20, -84528996754470930223e-20, -72299594230844519839e-20,
             -57679398168141327066e-20, -41683730779892996801e-20, -25514627335790291149e-20, -10710838211747769681e-20,
             12720145729326415607e-21, 14540842218988328389e-20, 33552500235752414908e-20, 60838109964484063119e-20]
        w = [31765161579790701148e-21, 71927618746964313778e-21, 10742555378156694842e-20, 13578811351554214795e-20,
             15492042553417744038e-20, 16300300254834219520e-20, 15784577013790806216e-20, 12921482926208917372e-20,
             46096943233133302568e-21, 20030610755774790850e-22, 95165705752725893549e-25, 40143360822128708729e-28]
    elif N == 13:
        x = [-98903182721370020265e-20, -94288936524363459773e-20, -86261843870640242196e-20, -75277808759167753869e-20,
             -61972590294795871779e-20, -47139332563986024748e-20, -31718188942187627557e-20, -16854863011308355787e-20,
             -41195843159851553906e-21, 71957380142115164738e-21, 22223926926874000328e-20, 42682885634093164862e-20,
             71270930856714354732e-20]
        w = [28069991026027589482e-21, 63803895087070663653e-21, 95973484361405430270e-21, 12264378189747678145e-20,
             14213612346123977130e-20, 15296686007570952707e-20, 15358437552921000921e-20, 14007635729175637795e-20,
             87531230524252970103e-21, 12989730151883234012e-21, 22351943999969127535e-23, 65097139765619073344e-26,
             18257341724040876662e-29]
    elif N == 14:
        x = [-99021130855943209687e-20, -94895368426058288869e-20, -87686856465753704289e-20, -77752669471002194917e-20,
             -65594116901081876554e-20, -51841232227159879604e-20, -37243750660439082187e-20, -22693429290756856295e-20,
             -93940943648510570987e-21, 16521198218716065629e-21, 13919799114797561344e-20, 30521886852802066309e-20,
             52192337126752562221e-20, 81957965081548293179e-20]
        w = [25060310888021301605e-21, 57137272611562033779e-21, 86434450014324433897e-21, 11141118228632175288e-20,
             13070790263291078499e-20, 14310195071194851995e-20, 14737968606274298328e-20, 14154903694980505066e-20,
             11456160782223814050e-20, 40466499493397342820e-21, 21701008894932486895e-22, 19960253076851250807e-24,
             39376501060604877095e-27, 76596142918862399780e-31]
    elif N == 15:
        x = [-99118619138431485634e-20, -95398089203095832045e-20, -88874665207045485764e-20, -79832886799647722652e-20,
             -68674462209286747178e-20, -55907326778454372362e-20, -42138595122137487519e-20, -28083407355763995168e-20,
             -14649293944496725019e-20, -30865949117072113052e-21, 75989566859912966734e-21, 21425891814116860148e-20,
             39280262275215780450e-20, 62012182191671475949e-20, 92858877219218103945e-20]
        w = [22570991165870390473e-21, 51589746641923392000e-21, 78401918844466166239e-21, 10176234626640128024e-20,
             12055819130110177262e-20, 13377324647273569326e-20, 14041818603247829422e-20, 13919569003129657925e-20,
             12562361445602688222e-20, 74852662340708470150e-21, 10996744175647251144e-21, 25513307315040157893e-23,
             15270418102934789627e-25, 21560859319293022163e-28, 30032040385910287756e-32]
    elif N == 16:
        x = [-99200289748411473927e-20, -95820266378296634182e-20, -89876661129475763142e-20, -81599671254865832971e-20,
             -71315812647978444249e-20, -59440032425488487666e-20, -46470396871945791541e-20, -32991653294098863600e-20,
             -19716091326862980561e-20, -76605243443508959615e-21, 26155046503992069925e-21, 14307776307824938682e-20,
             29506185654032182160e-20, 48403577800553841578e-20, 72091584865612160132e-20, 10394188783181811718e-19]
        w = [20484388078614008045e-21, 46916532350372347409e-21, 71569877291069983495e-21, 93424466379672137196e-21,
             11156011364306951297e-20, 12512553084306063601e-20, 13329704953113185969e-20, 13510959073859290681e-20,
             12840858805365359846e-20, 10016528657871746742e-20, 32102655847303900301e-21, 18115418480524121495e-22,
             24274994772381143993e-24, 10371321943363515335e-26, 10868941709467004901e-29, 11117372791599461059e-33]
        
    return np.array(x), np.array(w)

# Function that creates an energy mesh, including integration weights, to be used for
# integrals that are required to get observables
# The final mesh depends on the choice of T
def get_int_mesh(T: float, N_1: int, N_2: int, N_3: int, J: int, E_min: float, 
                 E_max: float, k_B: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:


    # In the case of zero temperature, we get an up->right->down contour
    if T < 1e-8:
        # We set a small T value to get a non-zero η value
        T_eff = 5e-4
        η = np.pi*k_B*T_eff
        
        N_tot = N_1+N_2+N_3
        Es = np.zeros((N_tot), dtype=np.complex128)
        Ws = np.zeros((N_tot), dtype=np.complex128)
        
        # 1. Get nodes and weights for up line
        x, w = GAULEG(N_1)
        # Transform range to [E_min, E_min+2i*J*η]
        δε = (0+1j)*J*η
        Es[:N_1] = x*δε + δε + E_min
        Ws[:N_1] = w*δε

        # 2. Get nodes and weights for right line
        x, w = GAULEG(N_2)
        # Transform range to [E_min+2i*J*η, E_max+2i*J*η]
        δε = 0.5*(E_max-E_min)
        Es[N_1:N_1+N_2] = x*δε + δε + E_min + (0+1j)*2.0*J*η
        Ws[N_1:N_1+N_2] = w*δε

        # 3. Get nodes and weights for down line
        x, w = GAULEG(N_3)
        x, w = x[::-1], w[::-1]
        # Transform range to [E_max+2i*J*η, E_max]
        δε = (0+1j)*J*η
        Es[N_1+N_2:] = x*δε + δε + E_max
        Ws[N_1+N_2:] = -w*δε
        
    # In the case of T > 0, the contour changes to up->right->right, see docs
    else:
        
        η = np.pi*k_B*T

        N_tot = N_1+N_2+N_3+J
        Es = np.zeros((N_tot), dtype=np.complex128)
        Ws = np.zeros((N_tot), dtype=np.complex128)

        # 1. Get nodes and weights for up line
        x, w = GAULEG(N_1)
        # Transform range to [E_min, E_min+2i*J*η]
        δε = (0+1j)*J*η
        Es[:N_1] = x*δε + δε + E_min
        Ws[:N_1] = w*δε

        # 2. Get nodes and weights for first right line
        x, w = GAULEG(N_2)
        # Transform range to [E_min+2i*J*η, E_max - 30k_BT + 2i*J*η]
        δε = 0.5*(E_max - E_min - 30.0*k_B*T)
        Es[N_1:N_1+N_2] = x*δε + δε + E_min + (0+1j)*2.0*J*η
        Ws[N_1:N_1+N_2] = w*δε
        
        # 3. Get nodes and weights for second right line
        x, w = GAUFD(N_3)
        # Transform range to [E_max - 30k_BT + 2i*J*η, E_max + 30k_BT + 2i*J*η]
        δε = 30.0*k_B*T
        Es[N_1+N_2:N_1+N_2+N_3] = x*δε + E_max + (0+1j)*2.0*J*η
        Ws[N_1+N_2:N_1+N_2+N_3] = w*δε

        # 4. Also add the contribution from the poles @ the Matsubara frequencies
        # These do not correspond to integration points ! We just incorporate them here
        x = np.arange(2*J-1, 0, -2)
        Es[N_1+N_2+N_3:] = E_max + (0+1j)*x*η
        Ws[N_1+N_2+N_3:] = -(0+1j)*2.0*η
    
    return Es, Ws