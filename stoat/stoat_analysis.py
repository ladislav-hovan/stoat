### Imports ###
from stoat.clustering import *

from typing import Optional, Literal, Dict, Any

### Class definition ###
class StoatAnalysis:
    def __init__(
        self,
        expr_df: pd.DataFrame,
        ind_df: pd.DataFrame,
        spatial_df: pd.DataFrame,
        validity: str = 'Valid',
        expr_clusters: Optional[pd.Series] = None,
        ind_clusters: Optional[pd.Series] = None,
    ) -> None:
        
        self.expression = expr_df
        self.indegrees = ind_df
        self.spatial = spatial_df
        self.validity = validity
        self.expr_clust = expr_clusters
        self.ind_clust = ind_clusters
    

    def determine_clusters(
        self,
        on_df: Literal['expr', 'ind', 'both'] = 'both',
        clust_settings: Dict[Any, Any] = {},
    ) -> None:
        
        if on_df in ['expr', 'both']:
            classes, ordering, n_classes = determine_cluster_labels(
                self.expression, self.spatial, self.validity, **clust_settings)
            self.classes_expr = classes
            self.ordering_expr = ordering
            self.n_classes_expr = n_classes
        if on_df in ['ind', 'both']:
            classes, ordering, n_classes = determine_cluster_labels(
                self.indegrees, self.spatial, self.validity, **clust_settings)
            self.classes_ind = classes
            self.ordering_ind = ordering
            self.n_classes_ind = n_classes


    