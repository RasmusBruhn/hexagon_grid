use super::{
    N,
    types::Index,
};

/// Defines a cyclic map of a single chunk
#[derive(Clone, Debug)]
pub struct MapCyclic<T: Tile> {
    /// The chunk
    chunk: Chunk<T>,
    /// The three edges, the ones opposite of each other are the same
    edges: [ChunkEdge<T>; 3],
    /// The three vertices, the ones opposite of each other are the same
    vertices: [ChunkVertex<T>; 3],
}

impl MapCyclic<TileTest> {
    pub fn new_layered() -> Self {
        Self {
            chunk: Chunk::<TileTest>::new_layered(),
            edges: std::array::from_fn(|_| ChunkEdge::<TileTest>::new_layered()),
            vertices: std::array::from_fn(|_| ChunkVertex::<TileTest>::new_layered()),
        }
    }
}

impl<T: Tile> Map for MapCyclic<T> {
    type T = T;

    fn get_chunk(&self, _index: Index) -> &Chunk<T> {
        &self.chunk
    }
    fn get_edge_vertical(&self, _index: Index) -> &ChunkEdge<T> {
        &self.edges[0]
    }
    fn get_edge_left(&self, _index: Index) -> &ChunkEdge<T> {
        &self.edges[1]
    }
    fn get_edge_right(&self, _index: Index) -> &ChunkEdge<T> {
        &self.edges[2]
    }
    fn get_vertex_bottom(&self, _index: Index) -> &ChunkVertex<T> {
        &self.vertices[0]
    }
    fn get_vertex_top(&self, _index: Index) -> &ChunkVertex<T> {
        &self.vertices[1]
    }
}

pub trait Map {
    type T: Tile;

    fn get_chunk(&self, index: Index) -> &Chunk<Self::T>;
    fn get_edge_vertical(&self, index: Index) -> &ChunkEdge<Self::T>;
    fn get_edge_left(&self, index: Index) -> &ChunkEdge<Self::T>;
    fn get_edge_right(&self, index: Index) -> &ChunkEdge<Self::T>;
    fn get_vertex_bottom(&self, index: Index) -> &ChunkVertex<Self::T>;
    fn get_vertex_top(&self, index: Index) -> &ChunkVertex<Self::T>;
}

/// Defines a chunk of hexagons in a hexagonal pattern
#[derive(Clone, Debug)]
pub struct Chunk<T: Tile> {
    /// The tiles of the hexagon
    tiles: [T; 3 * N * (N - 1) + 1],
}

impl<T: Tile> Chunk<T> {
    pub fn get_id(&self) -> Vec<u32> {
        self.tiles.iter().map(|tile| tile.get_id()).collect()
    }
}

impl Chunk<TileTest> {
    /// Creates a new chunk where each layer has its own colour
    fn new_layered() -> Self {
        let mut id = 0;
        let mut pos = 0;
        let tiles: [TileTest; 3 * N * (N - 1) + 1] = std::array::from_fn(move |_| {
            let tile = TileTest::new(id);
            pos += 1;
            if pos >= 6 * id {
                id += 1;
                pos = 0;
            }           
            tile
        });

        Self {
            tiles,
        }
    }
}

/// Defines an edge of a hexagonal pattern
#[derive(Clone, Debug)]
pub struct ChunkEdge<T: Tile>  {
    /// The tiles of the edge
    tiles: [T; N - 1],
}

impl<T: Tile> ChunkEdge<T> {
    pub fn get_id(&self) -> Vec<u32> {
        self.tiles.iter().map(|tile| tile.get_id()).collect()
    }
}

impl ChunkEdge<TileTest> {
    /// Creates a new edge for a layered chunk
    fn new_layered() -> Self {
        let tiles = std::array::from_fn(|_| TileTest::new(N as u32));

        Self {
            tiles,
        }
    }
}

/// Defines a vertex of a hexagonal pattern
#[derive(Clone, Debug)]
pub struct ChunkVertex<T: Tile>  {
    /// The tiles of the edge
    tile: T,
}

impl<T: Tile> ChunkVertex<T> {
    pub fn get_id(&self) -> Vec<u32> {
        vec![self.tile.get_id()]
    }
}

impl ChunkVertex<TileTest> {
    /// Creates a new edge for a layered chunk
    fn new_layered() -> Self {
        let tile = TileTest::new(N as u32);

        Self {
            tile,
        }
    }
}

/// Stores all the information about a single hexagonal tile
#[derive(Clone, Copy, Debug)]
pub struct TileTest {
    /// The id of this tile, this decides where the tile is located within the chunk
    id: u32,
}

impl TileTest {
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

impl Tile for TileTest {
    fn get_id(&self) -> u32 {
        self.id
    }
}

pub trait Tile: Clone + Copy {
    /// Retrieves the colour id of the tile
    fn get_id(&self) -> u32;    
}