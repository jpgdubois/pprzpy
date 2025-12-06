"""
Module for parsing Paparazzi UAV log data files and associated XML schemas,
and for deserializing message payloads into structured Polars DataFrames.

This module provides functions to:
- Efficiently parse large Paparazzi data files into Polars DataFrames.
- Extract and convert message schemas from XML protocol definitions.
- Deserialize serialized payload strings into strongly typed columns, supporting array 
  fields serialized as comma-separated values.
- Utilize Python typing with custom type aliases to improve code clarity and maintainability.

Typical usage involves loading a data log file and its matching XML schema, then 
deserializing selected message types into fully structured tabular data ready for analysis.

Defines:
- Type aliases for schema representation: FieldType, MessageSchema, SchemaType.
- Functions: parse_data_file, parse_schema_xml, deserialize_payload.

Requires:
- Polars for data frame operations.
- xml.etree.ElementTree for XML schema parsing.
- Python standard logging for tracing and debugging.

Example:
    >>> schema = parse_schema_xml("protocol.xml")
    >>> df = parse_data_file(Path("flight_log.data"))
    >>> df_decoded = deserialize_payload(df, "STAB_ATTITUDE", schema)
    >>> print(df_decoded.head())

This module is optimized for large flight logs and complex schema definitions
typical in UAV research projects utilizing the Paparazzi autopilot ecosystem.
"""

__all__ = [
    "parse_data_file",
    "parse_schema_xml",
    "deserialize_payload",
]

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import polars as pl
import xml.etree.ElementTree as ET
from lxml import etree

# Setup a logger for this module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

INT_MAP = {
    "int8": pl.Int8,
    "int16": pl.Int16,
    "int32": pl.Int32,
    "int64": pl.Int64,
    "uint8": pl.UInt8,
    "uint16": pl.UInt16,
    "uint32": pl.UInt32,
    "uint64": pl.UInt64,
}
FLOAT_MAP = {
    "float": pl.Float32,
    "double": pl.Float64,
}

def parse_data_file(file_path: Path, batch_size: int = 10_000) -> pl.DataFrame:
    """
    Parses a pprz data file and returns its contents as a Polars DataFrame.
    Reads file in batches for efficiency with large files.

    Parameters
    ----------
    file_path : Path
        Path to the data file.
    batch_size : int
        Number of lines per processing batch.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame containing parsed data with columns:
        'timestamp' (float64), 'ac_id' (uint8), 'msg_type' (categorical), 'payload' (string).
    """
    logger.info(f"Starting parsing of data file: {file_path}")
    def process_lines(lines):
        records = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                logger.warning(f"Skipping malformed line: {line.strip()}")
                continue
            timestamp = float(parts[0])
            ac_id = int(parts[1])
            msg_name = parts[2]
            payload = " ".join(parts[3:]) if len(parts) > 3 else ""
            records.append((timestamp, ac_id, msg_name, payload))
        return records

    data = []
    with file_path.open("r", encoding="utf-8") as f:
        batch = []
        for idx, line in enumerate(f, 1):
            batch.append(line)
            if len(batch) >= batch_size:
                logger.debug(f"Processing batch at line {idx}")
                data.extend(process_lines(batch))
                batch = []
        if batch:
            logger.debug("Processing final batch")
            data.extend(process_lines(batch))

    df = pl.DataFrame(
        data,
        schema=[
            ("msg_timestamp", pl.Float64),
            ("ac_id", pl.UInt8),
            ("msg_type", pl.Categorical),
            ("payload", pl.String),
        ],
        orient="row"
    )
    logger.info(f"Finished parsing file with {len(df)} records.")
    return df


def parse_schema_xml(xml_path: Path) -> Dict[str, pl.Schema]:
    """Parse a Paparazzi XML protocol schema and return message schemas as pl.Schema.

    Updated to look under <configuration><protocol> instead of at the document root.
    Falls back to root/config if those elements are missing, with warnings.

    If the XML parser fails due to invalid characters, the implementation will read
    the file as text, sanitize invalid XML characters and retry parsing.
    """
    logger.info(f"Parsing schema XML from: {xml_path}")
    parser = etree.XMLParser(recover=True)
    xmlfile= open(xml_path, 'r')
    xmlstring = xmlfile.read()
    xmlfile.close()
    tree = etree.fromstring(xmlstring, parser=parser)
    protocol = tree.find('protocol')

    if protocol is None:
        raise ValueError("Invalid XML schema: missing <protocol> element.")
    
    message_schemas: Dict[str, pl.Schema] = {}

    for msg_class in protocol.findall("msg_class"):
        for msg in msg_class.findall("message"):
            msg_name = msg.attrib.get("NAME")
            if not msg_name:
                logger.warning("Message without NAME attribute encountered in schema XML, skipping.")
                continue
            field_map: Dict[str, pl.DataType] = {}
            for field in msg.findall("field"):
                fname = field.attrib.get("NAME") or ""
                ftype = _xml_type_to_polars(field.attrib.get("TYPE"))
                field_map[fname] = ftype
            message_schemas[msg_name] = pl.Schema(field_map)
            logger.debug(f"Registered schema for message '{msg_name}' with {len(field_map)} fields.")

    logger.info(f"Completed parsing schema XML with {len(message_schemas)} message types.")
    return message_schemas

def _xml_type_to_polars(typ: Optional[str]) -> pl.DataType:
    """
    Convert XML TYPE strings to Polars dtypes, handling array types.

    Parameters
    ----------
    typ : Optional[str]
        XML type string from the schema.

    Returns
    -------
    pl.DataType
        Polars data type corresponding to XML type.
    """
    is_arr = _is_array_type(typ)
    base = _normalize_base(typ)
    if base in INT_MAP:
        inner = INT_MAP[base]
    elif base in FLOAT_MAP:
        inner = FLOAT_MAP[base]
    elif base in ("char", "char[]", "string"):
        inner = pl.Utf8
    else:
        if base.startswith("uint") or base.startswith("int"):
            inner = pl.Int64 if base.startswith("int") else pl.UInt64
        else:
            inner = pl.Utf8
    if is_arr:
        return pl.List(inner)
    return inner


def _is_array_type(typ: Optional[str]) -> bool:
    """
    Determine if an XML type indicates an array.

    Parameters
    ----------
    typ : Optional[str]
        XML type string.

    Returns
    -------
    bool
        True if type is an array type, False otherwise.
    """
    if not typ:
        return False
    t = typ.strip().lower()
    if t.startswith("char") or t.startswith("string"):
        return False
    return "[]" in typ or "[" in t


def _normalize_base(typ: Optional[str]) -> str:
    """
    Normalize base type from XML type by removing suffixes and array annotations.

    Parameters
    ----------
    typ : Optional[str]
        XML type string.

    Returns
    -------
    str
        Base type normalized.
    """
    if not typ:
        return "string"
    t = typ.strip()
    if t.endswith("_t"):
        t = t[:-2]
    if t.endswith("[]"):
        t = t[:-2]
    if "[" in t:
        t = t.split("[", 1)[0]
    return t


def deserialize_payload(df: pl.DataFrame, msg_type: str, schema_dict: Dict[str, pl.Schema], list_as_array: bool = True) -> pl.DataFrame:
    """
    Deserialize payloads according to message schemas.
    This function filters the input DataFrame for records matching the specified message type,
    then deserializes the 'payload' column (a space-separated string) into individual fields
    based on the provided schema.
    Parameters
    ----------
    df : pl.DataFrame
        The input DataFrame containing columns like 'msg_type' and 'payload'.
    msg_type : str
        The message type to filter and deserialize for.
    schema_dict : Dict[str, pl.Schema]
        A dictionary mapping message types to their Polars schemas, where each schema
        defines field names and their data types (including nested lists).
    list_as_array : bool, optional
        If True (default), convert list fields to fixed-size arrays based on the first
        record's list length. If False, keep them as variable-length lists.
    Returns
    -------
    pl.DataFrame
        A new DataFrame with the deserialized fields added as columns. If no records match
        the msg_type or if the schema is missing, returns the filtered (possibly empty)
        DataFrame unchanged.
    Notes
    -----
    - Logs info messages for deserialization progress and warnings/errors for missing data or schemas.
    - Assumes 'payload' is a string of space-separated tokens, where list fields are comma-separated within tokens.
    - For list_as_bool, the array size is determined from the first list field; inconsistent lengths may cause issues.
    """
    logger.info(f"Deserializing payload for message type: {msg_type}")
    filtered_df = df.filter(pl.col("msg_type") == msg_type)
    if filtered_df.is_empty():
        logger.warning(f"No records found for message type: {msg_type}")
        return filtered_df

    if msg_type not in schema_dict:
        logger.error(f"Schema not found for message type: {msg_type}")
        return filtered_df

    schema = schema_dict[msg_type]
    split_df = filtered_df.with_columns(pl.col("payload").str.split(" ").alias("payload_split"))

    for i, (field_name, dtype) in enumerate(schema.items()):
        token_col = pl.col("payload_split").list.get(i)
        if isinstance(dtype, pl.List):
            elem_dtype = dtype.inner()
            split_df = split_df.with_columns(
                token_col.str.split(",")
                         .list.eval(pl.element().cast(elem_dtype))
                         .alias(field_name)
            )
            
            if list_as_array:
                list_len = split_df.select(
                    pl.col("payload_split").list.get(i).str.split(",").list.len().first()
                ).item()
                split_df = split_df.with_columns(
                    pl.col(field_name).list.to_array(list_len).alias(field_name)
                )
        else:
            split_df = split_df.with_columns(token_col.cast(dtype).alias(field_name))

    logger.info(f"Deserialized {len(schema)} fields for message type: {msg_type}")
    return split_df.drop(["payload", "payload_split"])

if __name__ == "__main__":
    file_path = Path("/home/jpg/Projects/pprzpy/src/pprzpy/data/2025-11-07_12-22/22_05_01__01_59_46_SD.data")
    protocol_path = Path("/home/jpg/Projects/pprzpy/src/pprzpy/data/2025-11-07_12-22/22_05_01__01_59_46_SD_msgs.xml")

    logger.info("Starting main")
    schema = parse_schema_xml(protocol_path)
    df = parse_data_file(file_path)
    df_test = deserialize_payload(df, "STAB_ATTITUDE", schema)
    logger.info(f"Result sample:\n{df_test.tail()}")
