from __future__ import annotations
from typing import List, Union, TypeVar, Generic

import cupy as cp

T = TypeVar('T')

class TreeNode:
    """Represents a single node in a tree structure.
    
    Attributes:
        parent: The index of the parent node. The root node has a parent index of -1.
        children: A list of indices of the child nodes.
    """
    def __init__(self, parent: int, children: List[int]):
        self.parent = parent
        self.children = children

    def __repr__(self) -> str:
        return f"TreeNode(parent={self.parent}, children={self.children})"

    def find_child_idx(self, child_id: int) -> int:
        """Finds the local index of a child node."""
        try:
            return self.children.index(child_id)
        except ValueError:
            raise ValueError(f"Node {child_id} is not a child of this node.")

    def is_connected(self, other_id: int) -> bool:
        """Checks if this node is directly connected to another node."""
        return other_id == self.parent or other_id in self.children

    def find_bond(self, other_id: int) -> int:
        """Finds the bond index corresponding to a connection with another node.
        
        Bond 1 is reserved for the parent. Bonds 2, 3, ... correspond to
        the children in their respective order.
        """
        if other_id == self.parent:
            return 1
        try:
            # Child bond indices are 2-based
            return self.children.index(other_id) + 2
        except ValueError:
            raise ValueError(f"Node {other_id} is not connected to this node.")

class Tree:
    """Represents a tree as a collection of TreeNode objects.
    
    The tree structure is defined by parent-child relationships between nodes,
    which are stored as a list of TreeNode objects. Node indices correspond
    to their position in this list.
    
    Attributes:
        nodes: A list of TreeNode objects.
    """
    def __init__(self, nodes: List[TreeNode] | None = None):
        if nodes is None:
            # A new tree starts with a single root node with no parent (-1) and no children.
            self.nodes = [TreeNode(parent=-1, children=[])]
        else:
            self.nodes = nodes

    def __str__(self):
        def _str_helper(node_id):
            node = self.nodes[node_id]
            s = str(node_id)
            if len(node.children) > 0:
                s += " -> "
                if len(node.children) > 1:
                    s += "("
                s += ";".join(_str_helper(c) for c in node.children)
                if len(node.children) > 1:
                    s += ")"
            return s
        
        if not self.nodes:
            return "Tree([])"
        
        return _str_helper(self.find_head_node())

    def __repr__(self):
        return f"Tree(nodes={self.nodes!r})"

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)
    
    def __getitem__(self, i: Union[int, List[int], slice]) -> Union[TreeNode, List[TreeNode]]:
        if isinstance(i, (int, slice)):
            return self.nodes[i]
        elif isinstance(i, list):
            return [self.nodes[j] for j in i]
        raise TypeError(f"Invalid index type: {type(i)}")

    def __setitem__(self, i: int, v: TreeNode):
        self.nodes[i] = v

    def add_child(self, parent_id: int) -> Tree:
        """Adds a new child node to a specified parent node.
        
        Args:
            parent_id: The index of the node to which the new child will be added.
        
        Returns:
            The tree itself.
        """
        if not (0 <= parent_id < len(self)):
            raise IndexError(f"Parent ID {parent_id} out of bounds for tree of size {len(self)}")
        
        child_id = len(self)
        self.nodes.append(TreeNode(parent=parent_id, children=[]))
        self.nodes[parent_id].children.append(child_id)
        return self

    def add_children(self, parent_id: int, n: int) -> Tree:
        """Adds multiple child nodes to a specified parent node.
        
        Args:
            parent_id: The index of the node to which children will be added.
            n: The number of children to add.
            
        Returns:
            The tree itself.
        """
        for _ in range(n):
            self.add_child(parent_id)
        return self

    def add_tree(self, parent_id: int, other_tree: Tree) -> None:
        """Grafts another tree onto the current tree at a specified node.
        
        The entire `other_tree` is added as a new branch. The root of `other_tree`
        becomes a child of `parent_id`.
        
        Args:
            parent_id: The index of the node in this tree to attach to.
            other_tree: The tree to attach.
        
        Returns:
            None.
        """
        import copy
        other_tree_copy = copy.deepcopy(other_tree)
        
        if not other_tree_copy.nodes:
            return

        len_self = len(self)

        # Find the root of the tree being added
        other_root_idx_old = other_tree_copy.find_head_node()
        
        # All nodes from other_tree will be shifted by len_self
        # Create an index mapping from old to new
        idx_map = {i: i + len_self for i in range(len(other_tree_copy))}

        # The new index for the root of the other tree
        other_root_idx_new = idx_map[other_root_idx_old]

        # Attach the new subtree to the parent node
        self.nodes[parent_id].children.append(other_root_idx_new)

        # Re-index the nodes from other_tree before adding them
        for node in other_tree_copy.nodes:
            if node.parent == -1: # This is the root of the other tree
                node.parent = parent_id
            else:
                node.parent = idx_map[node.parent]
            
            node.children = [idx_map[c] for c in node.children]
        
        self.nodes.extend(other_tree_copy.nodes)

    def _get_subtree_nodes(self, node_id: int) -> List[int]:
        """Helper to get all node IDs in the subtree starting from node_id."""
        nodes_in_subtree = []
        nodes_to_visit = [node_id]
        visited = {node_id}
        while nodes_to_visit:
            curr_id = nodes_to_visit.pop(0)
            nodes_in_subtree.append(curr_id)
            for child_id in self.nodes[curr_id].children:
                if child_id not in visited:
                    nodes_to_visit.append(child_id)
                    visited.add(child_id)
        return nodes_in_subtree

    def remove_node(self, node_id: int) -> None:
        """Removes a node and its entire subtree recursively.
        
        This method removes the specified node and all of its descendants from the
        tree. The indices of the remaining nodes are updated accordingly.
        
        Args:
            node_id: The index of the node to remove.
        
        Returns:
            None.
        
        Raises:
            IndexError: If the node_id is out of bounds.
            ValueError: If an attempt is made to remove the root node.
        """
        self.loop_check()
        if not (0 <= node_id < len(self)):
            raise IndexError(f"Node ID {node_id} out of bounds for tree of size {len(self)}")

        parent_id = self.nodes[node_id].parent
        if parent_id == -1:
            raise ValueError("Attempt to remove the head-node.")

        for child_id in self.nodes[node_id].children[:]:
            self.remove_node(child_id)
            
        parent_node = self.nodes[parent_id]
        parent_node.children.remove(node_id)
        
        self.nodes.pop(node_id)
        
        # Re-index remaining nodes
        for i, node in enumerate(self.nodes):
            if node.parent > node_id:
                self.nodes[i].parent -= 1
            
            for j, child in enumerate(node.children):
                if child > node_id:
                    self.nodes[i].children[j] -= 1

    def set_head_node(self, new_head_id: int):
        """Sets a new root for the tree by restructuring parent-child relationships.
        
        This method re-roots the tree at `new_head_id`. The path from the old
        root to the new root is inverted.
        """
        if not (0 <= new_head_id < len(self)):
            raise IndexError("Invalid node id.")

        path_to_root = self.path_to_head(new_head_id)[:-1] # Exclude new_head_id itself
        
        for i in range(len(path_to_root) -1, -1, -1):
            current_node_id = path_to_root[i]
            parent_id = self[current_node_id].parent
            
            # Restructure tree
            self[parent_id].parent = current_node_id
            self[parent_id].children.remove(current_node_id)
            self[current_node_id].children.insert(0, parent_id)

        self[new_head_id].parent = -1

    def find_head_node(self) -> int:
        """Finds the root node of the tree.
        
        Returns:
            The index of the head (root) node.
            
        Raises:
            ValueError: If no head node is found.
        """
        for i, node in enumerate(self.nodes):
            if node.parent == -1:
                return i
        raise ValueError("No head node found in tree.")

    def bonds(self) -> List[tuple[int, int]]:
        """Returns a list of all bonds (parent-child connections) in the tree."""
        bond_list = []
        head_node_id = self.find_head_node()
        for i, node in enumerate(self.nodes):
            if i != head_node_id:
                bond_list.append((node.parent, i))
        return bond_list

    def adjacency_matrix(self) -> cp.ndarray:
        """Creates an adjacency matrix representation of the tree."""
        n = len(self)
        mat = cp.zeros((n, n), dtype=int)
        for parent, child in self.bonds():
            mat[parent, child] = 1
            mat[child, parent] = 1
        return mat
        
    def leaves(self) -> List[int]:
        """Identifies all leaf nodes in the tree.
        
        A node is considered a leaf if it has no children. A root node with
        exactly one child is also considered a leaf, which can be a convention
        for certain tensor network algorithms.
        """
        if not self.nodes:
            return []
        if len(self) == 1:
            return [0]

        leaf_nodes = []
        for i, node in enumerate(self.nodes):
            is_leaf = len(node.children) == 0
            # Per Julia implementation, root with one child is also a leaf.
            is_root_with_one_child = (len(node.children) == 1 and node.parent == -1)
            if is_leaf or is_root_with_one_child:
                leaf_nodes.append(i)
        return leaf_nodes
        
    def path_to_head(self, node_id: int) -> List[int]:
        """Returns the path from a given node to the head (root) of the tree."""
        path = [node_id]
        head_id = self.find_head_node()
        if node_id == head_id:
            return path
            
        curr_id = node_id
        while curr_id != head_id:
            parent_id = self.nodes[curr_id].parent
            if parent_id == -1 and curr_id != head_id:
                raise ValueError("Inconsistent parent structure found.")
            path.append(parent_id)
            curr_id = parent_id
        return path
        
    def path_from_head(self, node_id: int) -> List[int]:
        """Returns the path from the head (root) of the tree to a given node."""
        return self.path_to_head(node_id)[::-1]

    def loop_check(self):
        """Checks for loops or multiple parents in the tree structure."""
        try:
            head_node = self.find_head_node()
        except ValueError:
            raise ValueError("Tree has no head node, cannot check for loops.")
            
        seen_nodes = {head_node}
        for i in range(len(self)):
            for child in self.nodes[i].children:
                if child in seen_nodes:
                    raise ValueError(f"Loop detected: Node {child} has multiple parents or is part of a cycle.")
                seen_nodes.add(child)
        
        if len(seen_nodes) != len(self.nodes):
            raise ValueError("Tree contains disconnected components.")
            
        return True

    def path(self, start_node_id: int, end_node_id: int) -> List[int]:
        """Finds the unique path between two nodes in the tree."""
        if start_node_id is None:
            start_node_id = self.find_head_node()
        path_to_head_start = self.path_to_head(start_node_id)
        path_to_head_end = self.path_to_head(end_node_id)
        
        # Find the lowest common ancestor (LCA)
        path_to_head_end_set = set(path_to_head_end)
        lca = -1
        for node in path_to_head_start:
            if node in path_to_head_end_set:
                lca = node
                break
                
        if lca == -1:
            # This should not happen in a valid tree
            raise ValueError("Nodes are in disconnected components of the tree.")

        # Path from start_node to LCA
        start_to_lca = path_to_head_start[:path_to_head_start.index(lca)+1]
        
        # Path from LCA to end_node
        end_to_lca = path_to_head_end[:path_to_head_end.index(lca)]
        
        # The final path is from start to LCA, then from LCA to end (in reverse)
        return start_to_lca + end_to_lca[::-1]

    def plot(self, show_ids=True):
        """Plots the tree using networkx and matplotlib."""
        import networkx as nx
        import matplotlib.pyplot as plt

        if len(self) > 100:
            print("Tree is too large to plot, consider printing instead.")
            return

        adj_matrix = self.adjacency_matrix()
        # cupy -> numpy for networkx
        G = nx.from_numpy_array(adj_matrix.get())
        
        try:
            import pydot
            pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot', root=self.find_head_node())
        except ImportError:
            print("pydot not found, using a simpler layout. For a better tree layout, run `pip install pydot`.")
            pos = nx.spring_layout(G, seed=42)

        head_node = self.find_head_node()
        colors = ['#1f78b4'] * len(self)
        colors[head_node] = '#e31a1c'

        labels = {i: str(i) for i in range(len(self))} if show_ids else {}

        nx.draw(G, pos, with_labels=True, labels=labels, node_color=colors, node_size=500, arrows=False)
        plt.show()

    @classmethod
    def from_num_nodes(cls, num_nodes: int) -> Tree:
        """Creates a chain-like tree with a specified number of nodes."""
        if num_nodes <= 0:
            return cls(nodes=[])
        tree = cls()
        for i in range(num_nodes - 1):
            tree.add_child(i)
        return tree

    @staticmethod
    def random_tree(num_nodes: int, max_degree: int) -> Tree:
        """Constructs a random tree with a specified number of nodes and maximum degree."""
        import random
        if num_nodes <= 0:
            return Tree(nodes=[])
        
        tree = Tree()
        count = 1
        
        while count < num_nodes:
            added_in_pass = False
            current_leaves = tree.leaves()
            random.shuffle(current_leaves)

            for leaf_id in current_leaves:
                if count > 1 and leaf_id == 0:
                    continue

                current_degree = len(tree.nodes[leaf_id].children)
                if current_degree >= max_degree:
                    continue

                available_slots = max_degree - current_degree
                
                num_to_add = random.randint(1, available_slots)
                
                if count + num_to_add > num_nodes:
                    num_to_add = num_nodes - count
                
                if num_to_add > 0:
                    tree.add_children(leaf_id, num_to_add)
                    count += num_to_add
                    added_in_pass = True

                if count >= num_nodes:
                    break
            
            if not added_in_pass and count < num_nodes:
                non_full_nodes = [i for i, n in enumerate(tree.nodes) if len(n.children) < max_degree]
                if not non_full_nodes:
                    break
                
                node_to_add_to = random.choice(non_full_nodes)
                tree.add_child(node_to_add_to)
                count += 1

        return tree

class TreeNetwork(Generic[T]):
    """Associates a Tree with data at each node.
    
    This class combines a `Tree` structure with a list of 'sites', where each
    site holds data corresponding to a node in the tree. In our case, this
    data will typically be cuPy tensors.
    
    Attributes:
        tree: The underlying tree structure.
        sites: A list of data elements (e.g., tensors) corresponding to each node.
    """
    def __init__(self, tree: Tree, sites: List[T]):
        self.tree = tree
        self.sites = sites

    def __str__(self):
        return str(self.tree) + "\n" + str(self.sites)

    def __repr__(self):
        return f"TreeNetwork(tree={self.tree!r}, sites={self.sites!r})"

    @classmethod
    def from_sites(cls, sites: List[T]) -> TreeNetwork[T]:
        """Creates a TreeNetwork from a list of sites, with a chain-like tree."""
        tree = Tree.from_num_nodes(len(sites))
        return cls(tree, sites)
    
    def __len__(self) -> int:
        return len(self.tree.nodes)

    def __iter__(self):
        return iter(self.sites)

    def __getitem__(self, i: Union[int, List[int], slice]) -> Union[T, List[T]]:
        if isinstance(i, (int, slice)):
            return self.sites[i]
        elif isinstance(i, list):
            return [self.sites[j] for j in i]
        raise TypeError(f"Invalid index type: {type(i)}")

    def __setitem__(self, i: int, v: T):
        self.sites[i] = v

    def add_child(self, parent_id: int, site: T) -> None:
        """Adds a new child node and its associated site data.
        
        Args:
            parent_id: The index of the parent node.
            site: The data to associate with the new child node.
            
        Returns:
            None.
        """
        self.tree.add_child(parent_id)
        self.sites.append(site)

    def remove_node(self, node_id: int) -> None:
        """Removes a node and its data from the network."""
        # Loop over a copy of children because the tree is modified during iteration
        for child_id in self.tree[node_id].children[:]:
            self.remove_node(child_id)

        # Now that children are removed, remove the node itself
        self.tree.remove_node(node_id)
        self.sites.pop(node_id)

    def add_tree(self, parent_id: int, other_net: TreeNetwork[T]) -> None:
        """Grafts another TreeNetwork onto the current one."""
        self.tree.add_tree(parent_id, other_net.tree)
        self.sites.extend(other_net.sites)

    def set_head_node(self, new_head_id: int):
        """Sets a new root for the TreeNetwork.
        
        This involves restructuring the underlying tree and performing necessary
        tensor manipulations (reshape, transpose) on the sites to maintain
        consistency.
        """
        if not (0 <= new_head_id < len(self)):
            raise IndexError(f"Node ID {new_head_id} is out of bounds for TreeNetwork")

        current_head_id = self.tree.find_head_node()
        if new_head_id == current_head_id:
            return

        # Reshape current head to remove dummy dimension
        A_current_head = self.sites[current_head_id]
        if A_current_head.ndim > 0 and A_current_head.shape[0] == 1:
             self.sites[current_head_id] = A_current_head.reshape(A_current_head.shape[1:], order='F')

        path = self.tree.path_to_head(new_head_id) # Path from new_head up to current_head
        
        # Iterate from the node just below the root down to the new_head
        # This mirrors the Julia recursive implementation's order of operations
        for i in range(len(path) - 2, -1, -1):
            parent_id = path[i+1]
            child_id = path[i]
            
            A_parent = self.sites[parent_id]
            
            child_pos_in_parent = self.tree[parent_id].children.index(child_id)
            
            # Tensor axes are: (parent, child1, child2, ..., physical_indices...)
            # The bond to child_id is at axis child_pos_in_parent + 1
            # This bond will become the new parent bond (axis 0) for the tensor at parent_id
            # The original parent bond (axis 0) will become the first child bond.
            
            perm = [child_pos_in_parent + 1] + [0]
            
            # Add other children
            other_children_axes = [j + 1 for j in range(len(self.tree[parent_id].children)) if j != child_pos_in_parent]
            perm.extend(other_children_axes)

            # Append remaining dims (e.g. physical)
            num_bond_dims = len(self.tree[parent_id].children) + 1
            perm.extend(range(num_bond_dims, A_parent.ndim))

            self.sites[parent_id] = A_parent.transpose(perm)

        # Now that tensors are permuted, update the tree structure
        self.tree.set_head_node(new_head_id)
        
        # Add dummy dimension to new head
        A_new_head = self.sites[new_head_id]
        self.sites[new_head_id] = A_new_head.reshape((1,) + A_new_head.shape, order='F')

def find_head_node(tree: Union[Tree, TreeNetwork]) -> int:
    """Returns the ID of the head (root) node."""
    if isinstance(tree, TreeNetwork):
        return tree.tree.find_head_node()
    return tree.find_head_node()

def loop_check(tree: Union[Tree, TreeNetwork]) -> bool:
    """Checks the tree for loops or inconsistencies."""
    if isinstance(tree, TreeNetwork):
        tree = tree.tree
    tree.loop_check()
    return True

def path_from_head(tree: Tree, node_id: int) -> List[int]:
    """Finds the path from the head node to a given node."""
    return tree.path_from_head(node_id)

def find_child(node: TreeNode, child_id: int) -> int:
    """Finds the local index of a child."""
    return node.find_child_idx(child_id)

def get_leaves(tree: Tree) -> List[int]:
    """Returns a list of all leaf nodes."""
    return tree.leaves()

def get_bonds(tree: Union[Tree, TreeNetwork]) -> List[tuple[int, int]]:
    """Returns a list of all bonds (parent-child pairs)."""
    if isinstance(tree, TreeNetwork):
        return tree.tree.bonds()
    return tree.bonds()

def find_bond(node: TreeNode, other_id: int) -> int:
    """Finds the bond index connecting to another node."""
    return node.find_bond(other_id)

def set_head_node(net: Union[Tree, TreeNetwork], new_head_id: int) -> None:
    """Sets a new head (root) node for the tree, restructuring it."""
    if isinstance(net, TreeNetwork):
        # The TreeNetwork version of set_head_node handles tensors
        net.set_head_node(new_head_id)
    else:
        # The Tree version only restructures the tree
        net.set_head_node(new_head_id)

def path_to_head(tree: Tree, node_id: int) -> List[int]:
    """Finds the path from a given node to the head node."""
    return tree.path_to_head(node_id)