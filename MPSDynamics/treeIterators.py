import copy
from collections.abc import Sequence

from .treeBasics import Tree, TreeNetwork


class Path(Sequence):
    """
    An iterator and sequence that traces the unique path between two nodes in a tree.
    
    Examples:
        For a tree A -> (B, C) where B -> (D, E):
        
        >>> path = Path(tree, lastsite=4, firstsite=0)  # Path from A to E
        >>> list(path)  # Returns path: A -> B -> E
        [(node_A, 0, 0), (node_B, 1, 1), (node_E, 4, None)]
        
        >>> path[1]  # Get second node in path
        (node_B, 1, 1)
        
        >>> len(path)  # Length of path
        3
    """
    def __init__(self, iter_obj: Tree | TreeNetwork, lastsite: int, firstsite: int | None = None):
        if not isinstance(iter_obj, (Tree, TreeNetwork)):
            raise TypeError("iter_obj must be a Tree or TreeNetwork")
        self.iter_obj = iter_obj
        self._tree = iter_obj.tree if isinstance(iter_obj, TreeNetwork) else iter_obj
        
        self.firstsite = firstsite if firstsite is not None else self._tree.find_head_node()
        self.lastsite = lastsite
        self._path = self._tree.path(self.firstsite, self.lastsite)
        self._is_network = isinstance(self.iter_obj, TreeNetwork)

    def __iter__(self):
        for i, node_id in enumerate(self._path):
            try:
                next_node_id = self._path[i + 1]
                direction = self._tree[node_id].find_bond(next_node_id)
            except IndexError:
                direction = None
            
            if self._is_network:
                yield (self.iter_obj[node_id], node_id, direction)
            else:
                yield (self._tree[node_id], node_id, direction)

    @property
    def firstsite(self):
        return self._firstsite

    def __len__(self):
        return len(self._path)

    def __getitem__(self, i: int | slice):
        if isinstance(i, slice):
            indices = range(len(self))[i]
            return [self[j] for j in indices]

        if not isinstance(i, int):
            raise TypeError(f"Index must be an integer, not {type(i)}")

        if i < 0:
            i += len(self)

        if not (0 <= i < len(self)):
            raise IndexError("Index out of range")

        node_id = self._path[i]

        try:
            next_node_id = self._path[i + 1]
            direction = self._tree[node_id].find_bond(next_node_id)
        except IndexError:
            direction = None

        if self._is_network:
            return (self.iter_obj[node_id], node_id, direction)
        else:
            return (self._tree[node_id], node_id, direction)


class Radial:
    """
    Performs a breadth-first traversal of the tree, yielding nodes level by level.
    
    Examples:
        For a tree A -> (B, C) where B -> (D, E):
        
        >>> radial = Radial(tree)
        >>> list(radial)  # Returns levels: [A], [B, C], [D, E]
        [[0], [1, 2], [3, 4]]
        
        >>> radial[1]  # Get second level
        [1, 2]
        
        >>> len(radial)  # Number of levels
        3
    """
    def __init__(self, iter_obj: Tree | TreeNetwork, start: int | None = None):
        if not isinstance(iter_obj, (Tree, TreeNetwork)):
            raise TypeError("iter_obj must be a Tree or TreeNetwork")
        self.iter_obj = iter_obj
        self._tree = self.iter_obj.tree if isinstance(self.iter_obj, TreeNetwork) else self.iter_obj
        self.start = start if start is not None else self._tree.find_head_node()
        self._is_network = isinstance(self.iter_obj, TreeNetwork)

    def __iter__(self):
        if not self._tree.nodes:
            return

        current_level_indices = [self.start]
        if self._is_network:
            yield [(idx, self.iter_obj[idx]) for idx in current_level_indices]
        else:
            yield current_level_indices

        children = list(self._tree[self.start].children)

        while children:
            current_level_indices = list(children)
            if self._is_network:
                yield [(idx, self.iter_obj[idx]) for idx in current_level_indices]
            else:
                yield current_level_indices
            
            grandchildren = []
            for child_idx in children:
                grandchildren.extend(self._tree[child_idx].children)
            children = grandchildren

    def __len__(self):
        return sum(1 for _ in self)

    def __getitem__(self, i: int | slice):
        items = list(self)
        return items[i]


class Walk:
    """
    Performs a tree traversal, visiting each node, including parent nodes on backtrack.
    Yields: A, B, A, C, A for a tree A -> (B, C).
    """
    def __init__(self, iter_obj: Tree | TreeNetwork):
        if not isinstance(iter_obj, (Tree, TreeNetwork)):
            raise TypeError("iter_obj must be a Tree or TreeNetwork")
        self.iter_obj = iter_obj
        self._tree = self.iter_obj.tree if isinstance(self.iter_obj, TreeNetwork) else self.iter_obj
        self._is_network = isinstance(self.iter_obj, TreeNetwork)

    def __iter__(self):
        if not self._tree.nodes:
            return
            
        tree = copy.deepcopy(self._tree)
        state = tree.find_head_node()
        
        while state is not None:
            if self._is_network:
                yield (state, self.iter_obj[state])
            else:
                yield state

            node = tree[state]
            
            if node.children:
                state = node.children[0]
            else:
                parent_id = node.parent
                if parent_id == -1:
                    state = None
                else:
                    tree[parent_id].children.pop(0)
                    state = parent_id
    
    def __len__(self):
        return sum(1 for _ in self)

    def __getitem__(self, i: int | slice):
        items = list(self)
        return items[i]


class Traverse:
    """
    Performs a pre-order traversal of the tree (e.g., A, B, C for A -> (B,C)).
    
    Examples:
        For a tree A -> (B, C) where B -> (D, E):
        
        >>> traverse = Traverse(tree)
        >>> list(traverse)  # Returns pre-order: A, B, D, E, C
        [0, 1, 3, 4, 2]
        
        >>> traverse[2]  # Get third node in traversal
        3
        
        >>> len(traverse)  # Total number of nodes
        5
    """
    def __init__(self, iter_obj: Tree | TreeNetwork):
        if not isinstance(iter_obj, (Tree, TreeNetwork)):
            raise TypeError("iter_obj must be a Tree or TreeNetwork")
        self.iter_obj = iter_obj
        self._tree = self.iter_obj.tree if isinstance(self.iter_obj, TreeNetwork) else self.iter_obj
        self._is_network = isinstance(self.iter_obj, TreeNetwork)

    def __iter__(self):
        if not self._tree.nodes:
            return

        tree = copy.deepcopy(self._tree)
        state = tree.find_head_node()
        
        if self._is_network:
            yield (state, self.iter_obj[state])
        else:
            yield state

        while True:
            node = tree[state]
            
            par = state
            children = node.children

            while not children:
                par = tree[par].parent
                if par == -1:
                    return
                tree[par].children.pop(0)
                children = tree[par].children

            child_id = children[0]
            
            if self._is_network:
                yield (child_id, self.iter_obj[child_id])
            else:
                yield child_id

            state = child_id
            
    def __len__(self):
        return len(self.iter_obj)

    def __getitem__(self, i: int | slice):
        items = list(self)
        return items[i]


class Directional:
    """
    A wrapper for Walk and Traverse that yields the direction from the previous node.
    
    Examples:
        For a tree A -> (B, C) where B -> (D, E):
        
        >>> walk = Walk(tree)
        >>> directional = Directional(walk)
        >>> list(directional)  # Returns (direction, node) pairs
        [(0, 0), (0, 1), (1, 0), (1, 2), (1, 0)]
        
        >>> traverse = Traverse(tree)
        >>> directional = Directional(traverse)
        >>> list(directional)  # Returns (direction, node) pairs for pre-order
        [(None, 0), (0, 1), (0, 3), (1, 4), (1, 2)]
        
        >>> directional[1]  # Get second (direction, node) pair
        (0, 1)
        
        >>> len(directional)  # Total number of steps
        5
    """
    def __init__(self, iter_obj: Walk | Traverse):
        if not isinstance(iter_obj, (Traverse, Walk)):
             raise TypeError("iter_obj must be Traverse or Walk")
        self.iter_obj = iter_obj
        self._tree = iter_obj.iter_obj.tree if isinstance(iter_obj.iter_obj, TreeNetwork) else iter_obj.iter_obj
        self._is_network = isinstance(iter_obj.iter_obj, TreeNetwork)

    def __iter__(self):
        it = iter(self.iter_obj)

        try:
            current_val = next(it)
        except StopIteration:
            return

        while True:
            try:
                next_val = next(it)
                
                current_id = current_val[0] if self._is_network else current_val
                next_id = next_val[0] if self._is_network else next_val

                node = self._tree[current_id]
                
                if next_id == node.parent: # Moving to parent
                    direction = 1
                else: # Moving to a child
                    direction = node.find_bond(next_id)

                yield (direction, current_val)
                current_val = next_val

            except StopIteration:
                yield (None, current_val)
                return
    
    def __len__(self):
        return len(self.iter_obj)

    def __getitem__(self, i: int | slice):
        if isinstance(i, slice):
            items = list(self)
            return items[i]
        
        if not isinstance(i, int):
            raise TypeError("Index must be an integer")
    
        if i < 0:
            i += len(self)

        if not 0 <= i < len(self):
            raise IndexError("Index out of range")
        
        for j, item in enumerate(self):
            if j == i:
                return item
        
        raise IndexError("Index out of range") 