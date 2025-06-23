from .vina_docking import VinaDockingWrapper

def get_docking_engine(docking_engine_type: str='VinaDocking', **kwargs):
    if docking_engine_type == 'VinaDocking':
        return VinaDockingWrapper(**kwargs)
    else:
        raise ValueError(f"Invalid docking engine type: {docking_engine_type}")