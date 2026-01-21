"""
pk_energy.py
Pseudoknot energy parameters and scoring functions.

Based on extended energy model with pseudoknot-specific penalties:
- Pseudoknot initiation
- Band costs
- Unpaired bases in pseudoknots/bands
- Nested structures within pseudoknots
- Cross-band interactions
"""


class PKEnergyParams:
    """Pseudoknot energy parameters (in kcal/mol * 100 for Vienna compatibility)"""
    
    def __init__(self):
        # Core pseudoknot penalties
        self.Ps = 960    # Pseudoknot initiation penalty (9.6 kcal/mol)
        self.Psm = 1500  # Pseudoknot in multiloop (15.0 kcal/mol)
        self.Psp = 1500  # Nested pseudoknot inside pseudoknot (15.0 kcal/mol)
        self.Pb = 20     # Band penalty (0.2 kcal/mol per band)
        self.Pup = 10    # Unpaired base in pseudoknot/band (0.1 kcal/mol per base)
        self.Pps = 10    # Nested closed region inside pseudoloop (0.1 kcal/mol)
        
        # Cross-band scaling factors (multiplicative)
        self.stP = 0.83  # Stacking energy multiplier across bands
        self.intP = 0.83 # Interior loop energy multiplier across bands
        
        # Standard multiloop parameters (Vienna model)
        self.a = 340     # Multiloop offset (3.4 kcal/mol)
        self.b = 40      # Per-branch penalty in multiloop (0.4 kcal/mol)
        self.c = 0       # Per-unpaired in multiloop (0.0 kcal/mol)
        
        # Multiloop parameters when spanning bands (replace a, b, c)
        self.a_p = 340   # Multiloop offset when crossing band (3.4 kcal/mol)
        self.b_p = 40    # Per-branch penalty in band-crossing multiloop (0.4 kcal/mol)
        self.c_p = 0     # Per-unpaired in band-crossing multiloop (0.0 kcal/mol)


# Global instance
pk_params = PKEnergyParams()


def E_pk_initiation() -> int:
    """
    Energy penalty for initiating a pseudoknot.
    
    Returns:
        Energy in units of 10 cal/mol (for Vienna compatibility)
    """
    return pk_params.Ps


def E_band_cost(num_bands: int) -> int:
    """
    Energy cost for the number of bands in pseudoknot.
    
    Args:
        num_bands: Number of bands (typically 2 for H-type, more for complex)
    
    Returns:
        Energy in units of 10 cal/mol
    """
    # Pb penalty applies to each band structure
    return pk_params.Pb * num_bands


def E_unpaired_pk(num_unpaired: int) -> int:
    """
    Energy penalty for unpaired bases within pseudoknot or band.
    
    Args:
        num_unpaired: Number of unpaired bases
    
    Returns:
        Energy in units of 10 cal/mol
    """
    return pk_params.Pup * num_unpaired


def E_nested_in_pk(num_nested_regions: int) -> int:
    """
    Energy penalty for nested closed regions inside pseudoloop.
    
    Args:
        num_nested_regions: Number of nested structures
    
    Returns:
        Energy in units of 10 cal/mol
    """
    return pk_params.Pps * num_nested_regions


def E_pk_in_multiloop() -> int:
    """
    Energy penalty for pseudoknot inserted in multiloop.
    
    Returns:
        Energy in units of 10 cal/mol
    """
    return pk_params.Psm


def E_nested_pk() -> int:
    """
    Energy penalty for nested pseudoknot inside another pseudoknot.
    
    Returns:
        Energy in units of 10 cal/mol
    """
    return pk_params.Psp


def scale_stacking_cross_band(stack_energy: int) -> int:
    """
    Scale stacking energy when crossing band boundary.
    
    Args:
        stack_energy: Original stacking energy
    
    Returns:
        Scaled energy (reduced magnitude)
    """
    return int(stack_energy * pk_params.stP)


def scale_intloop_cross_band(intloop_energy: int) -> int:
    """
    Scale interior loop energy when crossing band boundary.
    
    Args:
        intloop_energy: Original interior loop energy
    
    Returns:
        Scaled energy (reduced magnitude)
    """
    return int(intloop_energy * pk_params.intP)


def E_multiloop_pk(num_branches: int, num_unpaired: int, crosses_band: bool = False) -> int:
    """
    Calculate multiloop energy, using band-crossing parameters if needed.
    
    Args:
        num_branches: Number of branches in multiloop
        num_unpaired: Number of unpaired bases
        crosses_band: Whether multiloop crosses a band
    
    Returns:
        Energy in units of 10 cal/mol
    """
    if crosses_band:
        # Use band-crossing multiloop parameters (a_p, b_p, c_p)
        a = pk_params.a_p
        b = pk_params.b_p
        c = pk_params.c_p
    else:
        # Use standard multiloop parameters (a, b, c)
        a = pk_params.a
        b = pk_params.b
        c = pk_params.c
    
    return a + b * num_branches + c * num_unpaired


def total_pk_energy(pk_type: str = 'H',
                    num_bands: int = 2,
                    num_unpaired: int = 0,
                    num_nested: int = 0,
                    in_multiloop: bool = False,
                    in_pk: bool = False) -> int:
    """
    Calculate total pseudoknot energy from components.
    
    Args:
        pk_type: Type of pseudoknot ('H', 'K', 'L', etc.)
        num_bands: Number of bands
        num_unpaired: Unpaired bases in pseudoknot
        num_nested: Nested structures inside
        in_multiloop: Whether PK is in a multiloop
        in_pk: Whether PK is nested in another PK
    
    Returns:
        Total energy in units of 10 cal/mol
    """
    energy = 0
    
    # Base initiation
    energy += E_pk_initiation()
    
    # Band costs
    energy += E_band_cost(num_bands)
    
    # Unpaired bases
    energy += E_unpaired_pk(num_unpaired)
    
    # Nested structures
    energy += E_nested_in_pk(num_nested)
    
    # Context penalties
    if in_multiloop:
        energy += E_pk_in_multiloop()
    
    if in_pk:
        energy += E_nested_pk()
    
    return energy


class PKBand:
    """Represents a band in a pseudoknot structure"""
    
    def __init__(self, start: int, end: int, band_id: int = 0):
        self.start = start
        self.end = end
        self.band_id = band_id
        self.pairs = []  # List of (i, j) base pairs in this band
        self.unpaired = 0
        self.energy = 0
    
    def add_pair(self, i: int, j: int):
        """Add a base pair to this band"""
        self.pairs.append((i, j))
    
    def crosses_band(self, other_band: 'PKBand') -> bool:
        """Check if this band crosses another band"""
        # Two bands cross if they have interleaved positions
        for i1, j1 in self.pairs:
            for i2, j2 in other_band.pairs:
                if (i1 < i2 < j1 < j2) or (i2 < i1 < j2 < j1):
                    return True
        return False
    
    def __repr__(self):
        return f"Band({self.start}-{self.end}, {len(self.pairs)} pairs)"


class PKStructure:
    """Represents a complete pseudoknot structure with energy calculation"""
    
    def __init__(self, pk_type: str = 'H'):
        self.pk_type = pk_type
        self.bands = []  # List of PKBand objects
        self.i = -1  # 5' position
        self.j = -1  # 3' position
        self.base_energy = 0  # Base pairing energies
        self.pk_penalty = 0   # Pseudoknot penalties
        self.total_energy = 0
    
    def add_band(self, band: PKBand):
        """Add a band to the pseudoknot"""
        self.bands.append(band)
    
    def calculate_energy(self, in_multiloop: bool = False, in_pk: bool = False):
        """
        Calculate total energy of this pseudoknot.
        
        Args:
            in_multiloop: Whether PK is in a multiloop context
            in_pk: Whether PK is nested in another PK
        """
        # Count unpaired bases across all bands
        total_unpaired = sum(band.unpaired for band in self.bands)
        
        # Count nested regions (simplified - would need full structure info)
        num_nested = 0
        
        # Calculate pseudoknot penalty
        self.pk_penalty = total_pk_energy(
            pk_type=self.pk_type,
            num_bands=len(self.bands),
            num_unpaired=total_unpaired,
            num_nested=num_nested,
            in_multiloop=in_multiloop,
            in_pk=in_pk
        )
        
        # Total energy = base pairing + PK penalties
        self.total_energy = self.base_energy + self.pk_penalty
    
    def __repr__(self):
        return f"PKStructure({self.pk_type}, {len(self.bands)} bands, E={self.total_energy/100:.2f} kcal/mol)"


def set_pk_parameters(Ps=None, Psm=None, Psp=None, Pb=None, Pup=None, Pps=None,
                      stP=None, intP=None, 
                      a=None, b=None, c=None,
                      a_p=None, b_p=None, c_p=None):
    """
    Set custom pseudoknot energy parameters.
    
    Args:
        Ps: Pseudoknot initiation (10 cal/mol)
        Psm: PK in multiloop (10 cal/mol)
        Psp: Nested PK penalty (10 cal/mol)
        Pb: Band cost (10 cal/mol per band)
        Pup: Unpaired in PK (10 cal/mol per base)
        Pps: Nested structure penalty (10 cal/mol)
        stP: Cross-band stacking multiplier (0-1)
        intP: Cross-band intloop multiplier (0-1)
        a: Standard multiloop offset (10 cal/mol)
        b: Standard multiloop branch cost (10 cal/mol)
        c: Standard multiloop unpaired cost (10 cal/mol)
        a_p: Multiloop offset for band-crossing (10 cal/mol)
        b_p: Multiloop branch cost for band-crossing (10 cal/mol)
        c_p: Multiloop unpaired cost for band-crossing (10 cal/mol)
    """
    global pk_params
    
    if Ps is not None: pk_params.Ps = Ps
    if Psm is not None: pk_params.Psm = Psm
    if Psp is not None: pk_params.Psp = Psp
    if Pb is not None: pk_params.Pb = Pb
    if Pup is not None: pk_params.Pup = Pup
    if Pps is not None: pk_params.Pps = Pps
    if stP is not None: pk_params.stP = stP
    if intP is not None: pk_params.intP = intP
    if a is not None: pk_params.a = a
    if b is not None: pk_params.b = b
    if c is not None: pk_params.c = c
    if a_p is not None: pk_params.a_p = a_p
    if b_p is not None: pk_params.b_p = b_p
    if c_p is not None: pk_params.c_p = c_p


def get_pk_parameters():
    """
    Get current pseudoknot energy parameters.
    
    Returns:
        Dictionary of parameters
    """
    return {
        'Ps': pk_params.Ps,
        'Psm': pk_params.Psm,
        'Psp': pk_params.Psp,
        'Pb': pk_params.Pb,
        'Pup': pk_params.Pup,
        'Pps': pk_params.Pps,
        'stP': pk_params.stP,
        'intP': pk_params.intP,
        'a': pk_params.a,
        'b': pk_params.b,
        'c': pk_params.c,
        'a_p': pk_params.a_p,
        'b_p': pk_params.b_p,
        'c_p': pk_params.c_p
    }
