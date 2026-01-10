import requests
import os
from datetime import datetime
from tqdm import tqdm
import zstandard as zstd
import chess.pgn
import chess
import numpy as np
from IPython.display import display, SVG


def download_file(url, filename):
    """
    Downloads a file from the specified URL with progress display.
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    # Check if the file exists
    if os.path.exists(filename):
        print(f"File {filename} already exists.")
        return

    # Download the file with progress
    with (
        open(filename, "wb") as f,
        tqdm(
            desc=filename,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"File {filename} successfully downloaded.")


def decompress_zst_file(input_file, output_file):
    """
    Decompresses a .zst file with progress display.
    """
    if os.path.exists(output_file):
        print(f"File {output_file} already exists.")
        return

    # Open the file for reading
    with open(input_file, "rb") as f:
        decompressor = zstd.ZstdDecompressor()
        total_size = os.path.getsize(input_file)

        # Create a streaming reader for decompression
        with (
            decompressor.stream_reader(f) as reader,
            open(output_file, "wb") as out,
            tqdm(
                desc=f"Decompressing {input_file}",
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            while True:
                chunk = reader.read(8192)
                if not chunk:
                    break
                out.write(chunk)
                bar.update(len(chunk))
    print(f"File {input_file} successfully decompressed to {output_file}.")


def download_lichess_dataset(year, month):
    """
    Downloads the Lichess dataset with games for the specified year and month.
    """
    # Form the URL for downloading
    url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"

    # Filename for saving
    filename = f"lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"

    # Download the file
    download_file(url, filename)

    # Decompress the file
    output_file = filename.replace(".zst", "")
    decompress_zst_file(filename, output_file)


def download_lichess_datasets_for_period(start_year, start_month, end_year, end_month):
    """
    Downloads datasets for the specified period.
    """
    current_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 1)

    while current_date <= end_date:
        download_lichess_dataset(current_date.year, current_date.month)
        # Move to the next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)


def read_pgn_and_convert_to_fen(year, month, move_offset=0):
    """
    Reads a PGN file and converts moves to FEN.
    """
    filename = f"lichess_db_standard_rated_{year}-{month:02d}.pgn"

    with open(filename, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            moves = list(game.mainline_moves())

            if len(moves) <= move_offset:
                continue

            for move in moves[: -move_offset - 1]:
                board.push(move)

            fen = board.fen()
            yield fen


# Dictionary for encoding pieces
chess_dict = {
    "p": 1,
    "P": 2,
    "n": 3,
    "N": 4,
    "b": 5,
    "B": 6,
    "r": 7,
    "R": 8,
    "q": 9,
    "Q": 10,
    "k": 11,
    "K": 12,
    ".": 0,  # Empty square
}


def fen_to_array(fen):
    """
    Converts FEN to a numpy array of size 8x8.
    :param fen: FEN string (only the part with piece positions).
    :return: numpy array 8x8, where each element is a number from 0 to 12.
    """
    # Create an empty 8x8 array
    board_array = np.zeros((8, 8), dtype=int)

    # Split FEN into rows
    fen = fen.split(" ")[0]
    rows = fen.split("/")

    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                # If the character is a digit, skip the specified number of squares
                for _ in range(int(char)):
                    board_array[i, col] = chess_dict["."]  # Empty square
                    col += 1
            else:
                # If the character is a piece, encode it
                board_array[i, col] = chess_dict[char]
                col += 1

    return board_array


def array_to_fen(board_array):
    """
    Converts a numpy array 8x8 to a FEN string.
    :param board_array: numpy array 8x8, where each element is a number from 0 to 12.
    :return: FEN string (part with piece positions).
    """
    fen_rows = []
    reverse_chess_dict = {v: k for k, v in chess_dict.items()}  # Reverse dictionary

    for i in range(8):
        row = []
        empty_count = 0

        for j in range(8):
            # Get the number for the current square
            num = board_array[i, j]

            if num == 0:
                # If the square is empty, increment the empty square counter
                empty_count += 1
            else:
                # If a piece is encountered, add the number of empty squares (if any)
                if empty_count > 0:
                    row.append(str(empty_count))
                    empty_count = 0
                row.append(reverse_chess_dict[num])

        # Add remaining empty squares at the end of the row
        if empty_count > 0:
            row.append(str(empty_count))

        # Assemble the row string and add to FEN
        fen_rows.append("".join(row))

    # Join rows with '/'
    fen = "/".join(fen_rows)
    return fen


def create_dataset_from_pgn(year, month, num_samples=None):
    """
    Creates a dataset from the last positions of chess games in a PGN file.
    :param pgn_file: Path to the PGN file.
    :param num_samples: Number of samples to create (if None, all games are processed).
    :return: numpy array with data of size (N, 8, 8, 13), where N is the number of samples.
    """
    dataset = []
    generator = read_pgn_and_convert_to_fen(year, month, 2)  # FEN strings generator

    for i, fen in enumerate(generator):
        if num_samples is not None and i >= num_samples:
            break  # Limit the number of samples

        # Convert FEN to numpy array
        try:
            board_array = fen_to_array(fen)
        except:
            print(fen)
        dataset.append(board_array)

        # Output progress
        if i % 10000 == 0:
            print(f"Processed {i} games")

    # Convert list to numpy array
    dataset = (np.array(dataset) - 6) / 6
    
    return dataset


def display_board(fen):
    """
    Visualizes a FEN position as a chess board.
    :param fen: FEN string.
    """
    board = chess.Board(fen)
    display(SVG(chess.svg.board(board=board, size=300)))
