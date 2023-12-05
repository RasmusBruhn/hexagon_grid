use super::types::Index;

/// Defines a cyclic map of a single chunk
#[derive(Clone, Debug)]
pub struct MapCyclic<T: Tile> {
    /// The size of the a chunk
    size: usize,
    /// The chunk
    chunk: Chunk<T>,
    /// The three edges
    edges: [Chunk<T>; 3],
    /// The two vertices
    vertices: [Chunk<T>; 2],
}

impl MapCyclic<TileID> {
    /// Creates a new cyclic map where the inner layer is id: 0 and each layer out increases the id by 1
    /// 
    /// # Parameters
    /// 
    /// size: The size of the chunk
    pub fn new_layered(size: usize) -> Self {
        Self {
            size,
            chunk: Chunk::<TileID>::new_layered_chunk(size),
            edges: std::array::from_fn(|_| Chunk::<TileID>::new_layered_edge(size)),
            vertices: std::array::from_fn(|_| Chunk::<TileID>::new_layered_vertex(size)),
        }
    }
}

impl<T: Tile> Map for MapCyclic<T> {
    type T = T;

    fn get_size(&self) -> usize {
        self.size
    }

    fn get_chunk(&self, _index: &Index) -> &Chunk<T> {
        &self.chunk
    }
    fn get_edge_vertical(&self, _index: &Index) -> &Chunk<T> {
        &self.edges[0]
    }
    fn get_edge_left(&self, _index: &Index) -> &Chunk<T> {
        &self.edges[1]
    }
    fn get_edge_right(&self, _index: &Index) -> &Chunk<T> {
        &self.edges[2]
    }
    fn get_vertex_bottom(&self, _index: &Index) -> &Chunk<T> {
        &self.vertices[0]
    }
    fn get_vertex_top(&self, _index: &Index) -> &Chunk<T> {
        &self.vertices[1]
    }
}

impl Chunk<TileID> {
    /// Creates a new chunk where the inner layer is id: 0 and each layer out increases the id by 1
    /// 
    /// # Parameters
    /// 
    /// size: The size of the chunk
    fn new_layered_chunk(size: usize) -> Self {
        let tiles = [TileID::new(0)]
            .into_iter()
            .chain((1..size)
            .map(|layer| {
                (0..6)
                    .map(|_| {
                        (0..layer)
                            .map(|_| {
                                TileID::new(layer as u32)
                            })
                            .collect::<Vec<TileID>>()
                            .into_iter()
                    })
                    .flatten()
                    .collect::<Vec<TileID>>()
            })
            .flatten())
            .collect();

        Self {
            tiles,
        }
    }

    /// Creates a new edge where the id's are all equal to size (consistent with new_layered_chunk)
    /// 
    /// # Parameters
    /// 
    /// size: The size of the chunk
    fn new_layered_edge(size: usize) -> Self {
        let tiles = (0..size - 1)
            .map(|_| TileID::new(size as u32))
            .collect();

        Self {
            tiles
        }
    }

    /// Creates a new vertex where the id is equal to size (consistent with new_layered_chunk)
    /// 
    /// # Parameters
    /// 
    /// size: The size of the chunk
    fn new_layered_vertex(size: usize) -> Self {
        let tiles = vec![TileID::new(size as u32)];

        Self {
            tiles
        }
    }
}

/// The simplest tile just storing an id
#[derive(Clone, Copy, Debug)]
pub struct TileID {
    /// The id of this tile
    id: u32,
}

impl TileID {
    /// Creates a new tile
    /// 
    /// # Parameters
    /// 
    /// id: The id of the tile
    fn new(id: u32) -> Self {
        Self {
            id,
        }
    }
}

impl Tile for TileID {
    fn get_id(&self) -> u32 {
        self.id
    }
}

/// Defines a map of chunks of hexagon tiles
pub trait Map {
    type T: Tile;

    /// Retrieves the size of a chunk
    fn get_size(&self) -> usize;

    /// Retrieves the chunk at the given index
    /// 
    /// # Parameters
    /// 
    /// # index: The index of the chunk
    fn get_chunk(&self, index: &Index) -> &Chunk<Self::T>;

    /// Retrieves the vertical edge (0) associated with the chunk at the given index
    /// 
    /// # Parameters
    /// 
    /// # index: The index of the associated chunk
    fn get_edge_vertical(&self, index: &Index) -> &Chunk<Self::T>;

    /// Retrieves the left edge (1) associated with the chunk at the given index
    /// 
    /// # Parameters
    /// 
    /// # index: The index of the associated chunk
    fn get_edge_left(&self, index: &Index) -> &Chunk<Self::T>;

    /// Retrieves the right edge (2) associated with the chunk at the given index
    /// 
    /// # Parameters
    /// 
    /// # index: The index of the associated chunk
    fn get_edge_right(&self, index: &Index) -> &Chunk<Self::T>;

    /// Retrieves the bottom vertex (0) associated with the chunk at the given index
    /// 
    /// # Parameters
    /// 
    /// # index: The index of the associated chunk
    fn get_vertex_bottom(&self, index: &Index) -> &Chunk<Self::T>;

    /// Retrieves the top vertex (1) associated with the chunk at the given index
    /// 
    /// # Parameters
    /// 
    /// # index: The index of the associated chunk
    fn get_vertex_top(&self, index: &Index) -> &Chunk<Self::T>;
}

/// Defines a chunk of hexagons in a hexagonal pattern, this may be used as a chunk, edge or vertex
#[derive(Clone, Debug)]
pub struct Chunk<T: Tile> {
    /// The tiles of the hexagon
    tiles: Vec<T>,
}

impl<T: Tile> Chunk<T> {
    /// Retrieves all of the ids of the chunk
    pub fn get_id(&self) -> Vec<u32> {
        self.tiles.iter().map(|tile| tile.get_id()).collect()
    }
}

pub trait Tile: Clone + Copy {
    /// Retrieves the colour id of the tile
    fn get_id(&self) -> u32;    
}