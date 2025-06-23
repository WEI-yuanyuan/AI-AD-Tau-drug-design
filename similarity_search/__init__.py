from .FastROCS import FastROCSSimilaritySearch

def get_similarity_search_engine(lib_path, type='FastROCS', **kwargs):
    if type == 'FastROCS':
        return FastROCSSimilaritySearch(lib_path, **kwargs)
    else:
        raise ValueError(f"Invalid search engine type: {type}")