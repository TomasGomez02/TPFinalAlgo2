import numpy as np
import pandas as pd
import numpy.typing
from scipy.sparse import spmatrix

ArrayLike = list | np.ndarray | pd.DataFrame
MatrixLike = np.ndarray | pd.DataFrame