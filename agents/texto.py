"""El texto de un mensaje, aunque el proveedor lo devuelva en bloques."""


def texto_de(contenido: object) -> str:
    """str, o la unión de los bloques {'type': 'text', 'text': ...}. Ignora metadata interna."""
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        partes = []
        for bloque in contenido:
            if isinstance(bloque, dict):
                partes.append(str(bloque.get("text", "")))
            else:
                partes.append(str(getattr(bloque, "text", bloque)))
        return "".join(partes)
    return str(contenido)
