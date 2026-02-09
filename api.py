from fastapi import FastAPI, HTTPException, Form, Request
from pydantic import BaseModel, Field
import re
from typing import Optional

app = FastAPI(
    title="API Segmenteur de Texte",
    description="API pour segmenter un texte en un nombre spécifique de segments",
    version="1.0.0"
)


class SegmentRequest(BaseModel):
    SCRIPT: str = Field(..., description="Le texte à segmenter")
    NUMBER_IMAGES: str = Field(..., description="Le nombre de segments souhaités")

    class Config:
        json_schema_extra = {
            "example": {
                "SCRIPT": "Ceci est une phrase. Voici une autre phrase! Et encore une dernière?",
                "NUMBER_IMAGES": "5"
            }
        }


class SegmentResponse(BaseModel):
    segments: str
    timestamps: str
    nombre_segments: int


def segmenter_texte(texte, nb_segments):
    """
    Segmente un texte en exactement nb_segments segments.
    Si le nombre de phrases dépasse nb_segments, fusionne les phrases les plus courtes.
    Si le nombre de phrases est inférieur à nb_segments, divise les phrases les plus longues.
    La division se fait à la virgule la plus centrée, SAUF si cela crée un segment de 4 mots ou moins,
    auquel cas on coupe au milieu du segment.

    Rééquilibrage final : si une virgule au milieu d'un segment sépare des parties contenant chacune
    2 fois plus de mots que le segment le plus court, on divise à cette virgule et on fusionne
    le segment court avec son voisin le plus court.

    Args:
        texte: Le texte à segmenter
        nb_segments: Le nombre de segments souhaités (défaut: 10)

    Returns:
        Un tuple (segments_str, timestamps_str) avec:
        - segments_str: segments numérotés (1.segment1 2.segment2 etc.)
        - timestamps_str: timestamps séparés par '/' (ou None si pas de timestamps)
    """

    # Parser le texte avec timestamps si présents
    if '/' in texte and texte.count('/') > 1:
        elements = texte.strip().split('/')
        mots = []
        timestamps = []
        for i in range(0, len(elements), 2):
            if i < len(elements):
                mots.append(elements[i])
            if i + 1 < len(elements):
                timestamps.append(elements[i + 1])
        texte_simple = ' '.join(mots)
    else:
        texte_simple = texte
        timestamps = None

    # Segmentation en phrases (gère . ! ? suivis d'espace ou fin de texte)
    phrases = re.split(r'(?<=[.!?])\s+', texte_simple.strip())
    phrases = [p.strip() for p in phrases if p.strip()]

    segments = phrases.copy()

    # Si on a moins de phrases que de segments souhaités, diviser les phrases les plus longues
    while len(segments) < nb_segments:
        # Trouver la phrase la plus longue
        max_longueur = 0
        max_index = 0

        for i in range(len(segments)):
            if len(segments[i]) > max_longueur:
                max_longueur = len(segments[i])
                max_index = i

        phrase = segments[max_index]

        # Chercher toutes les virgules dans la phrase
        virgules = [i for i, char in enumerate(phrase) if char == ',']

        coupe_valide = False

        if virgules:
            # Trouver la virgule la plus proche du centre
            milieu = len(phrase) // 2
            virgule_centrale = min(virgules, key=lambda x: abs(x - milieu))

            # Diviser à cette virgule (inclure la virgule dans la première partie)
            partie1 = phrase[:virgule_centrale + 1].strip()
            partie2 = phrase[virgule_centrale + 1:].strip()

            # Vérifier si les deux parties ont plus de 4 mots
            nb_mots_partie1 = len(partie1.split())
            nb_mots_partie2 = len(partie2.split())

            if nb_mots_partie1 > 4 and nb_mots_partie2 > 4:
                coupe_valide = True

        if not coupe_valide:
            # Pas de virgule valide, diviser au milieu sur un espace
            milieu = len(phrase) // 2
            pos_coupe = phrase.rfind(' ', 0, milieu)
            if pos_coupe == -1:
                pos_coupe = milieu

            partie1 = phrase[:pos_coupe].strip()
            partie2 = phrase[pos_coupe:].strip()

        # Remplacer la phrase par ses deux parties
        segments[max_index] = partie1
        segments.insert(max_index + 1, partie2)

    # Si on a trop de phrases, fusionner les plus courtes
    while len(segments) > nb_segments:
        # Trouver les deux phrases consécutives les plus courtes
        min_longueur = float('inf')
        min_index = 0

        for i in range(len(segments) - 1):
            longueur_combinee = len(segments[i]) + len(segments[i + 1])
            if longueur_combinee < min_longueur:
                min_longueur = longueur_combinee
                min_index = i

        # Fusionner les deux phrases
        segments[min_index] = segments[min_index] + ' ' + segments[min_index + 1]
        segments.pop(min_index + 1)

    # RÉÉQUILIBRAGE FINAL : vérifier les virgules centrales
    segments_modifies = True
    while segments_modifies:
        segments_modifies = False

        # Trouver le segment avec le moins de mots
        nb_mots_segments = [len(seg.split()) for seg in segments]
        min_mots = min(nb_mots_segments)

        # Chercher un segment avec une virgule centrale qui sépare des parties
        # contenant chacune au moins 2*min_mots
        for i in range(len(segments)):
            segment = segments[i]
            virgules = [j for j, char in enumerate(segment) if char == ',']

            if virgules:
                # Chercher la virgule la plus proche du centre
                milieu = len(segment) // 2
                virgule_centrale = min(virgules, key=lambda x: abs(x - milieu))

                partie1 = segment[:virgule_centrale + 1].strip()
                partie2 = segment[virgule_centrale + 1:].strip()

                nb_mots_partie1 = len(partie1.split())
                nb_mots_partie2 = len(partie2.split())

                # Si les deux parties ont au moins 2 fois plus de mots que le plus petit segment
                if nb_mots_partie1 >= 2 * min_mots and nb_mots_partie2 >= 2 * min_mots:
                    # Diviser ce segment
                    segments[i] = partie1
                    segments.insert(i + 1, partie2)

                    # Trouver le segment le plus court (excluant les deux parties qu'on vient de créer)
                    min_mots_index = -1
                    min_mots_count = float('inf')
                    for j in range(len(segments)):
                        if j != i and j != i + 1:
                            nb_mots = len(segments[j].split())
                            if nb_mots < min_mots_count:
                                min_mots_count = nb_mots
                                min_mots_index = j

                    # Fusionner le segment le plus court avec son voisin le plus court
                    if min_mots_index != -1:
                        if min_mots_index > 0 and min_mots_index < len(segments) - 1:
                            # Le segment a deux voisins
                            nb_mots_avant = len(segments[min_mots_index - 1].split())
                            nb_mots_apres = len(segments[min_mots_index + 1].split())

                            if nb_mots_avant <= nb_mots_apres:
                                # Fusionner avec le voisin avant
                                segments[min_mots_index - 1] = segments[min_mots_index - 1] + ' ' + segments[
                                    min_mots_index]
                                segments.pop(min_mots_index)
                            else:
                                # Fusionner avec le voisin après
                                segments[min_mots_index] = segments[min_mots_index] + ' ' + segments[min_mots_index + 1]
                                segments.pop(min_mots_index + 1)
                        elif min_mots_index == 0:
                            # Premier segment, fusionner avec le suivant
                            segments[0] = segments[0] + ' ' + segments[1]
                            segments.pop(1)
                        else:
                            # Dernier segment, fusionner avec le précédent
                            segments[min_mots_index - 1] = segments[min_mots_index - 1] + ' ' + segments[min_mots_index]
                            segments.pop(min_mots_index)

                    segments_modifies = True
                    break

    # Numéroter les segments
    segments_numerotes = [f"{i + 1}.{segment}" for i, segment in enumerate(segments)]

    # Générer les timestamps si présents
    if timestamps is not None:
        timestamps_segments = []
        position_mot = 0

        for segment in segments:
            mots_segment = segment.split()

            # Trouver le timestamp du premier mot du segment
            if position_mot < len(timestamps):
                timestamp_debut = timestamps[position_mot]
            else:
                timestamp_debut = "0.0"

            timestamps_segments.append(timestamp_debut)

            # Avancer la position
            position_mot += len(mots_segment)

        return ' '.join(segments_numerotes), '/'.join(timestamps_segments)
    else:
        return ' '.join(segments_numerotes), None


@app.get("/", tags=["Root"])
async def root():
    """Point d'entrée de l'API"""
    return {
        "message": "API Segmenteur de Texte",
        "endpoints": {
            "POST /segmenter": "Segmenter un texte (JSON ou form-data)",
            "POST /segmenter-json": "Segmenter un texte (JSON uniquement)",
            "POST /segmenter-form": "Segmenter un texte (form-data uniquement)",
            "GET /docs": "Documentation interactive"
        }
    }


@app.post("/segmenter-json", response_model=SegmentResponse, tags=["Segmentation"])
async def segmenter_json(request: SegmentRequest):
    """
    Segmente un texte - ACCEPTE UNIQUEMENT JSON

    - **SCRIPT**: Le texte à segmenter (peut contenir des timestamps)
    - **NUMBER_IMAGES**: Le nombre de segments souhaités

    Retourne le texte segmenté avec les segments numérotés (1.segment1 2.segment2 etc.)
    """
    try:
        nb_segments = int(request.NUMBER_IMAGES)

        if nb_segments <= 0:
            raise HTTPException(
                status_code=400,
                detail="NUMBER_IMAGES doit être un nombre positif"
            )

        resultat_segments, resultat_timestamps = segmenter_texte(request.SCRIPT, nb_segments)

        return SegmentResponse(
            segments=resultat_segments,
            timestamps=resultat_timestamps or "",
            nombre_segments=nb_segments
        )

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="NUMBER_IMAGES doit être une chaîne représentant un nombre entier valide"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la segmentation: {str(e)}"
        )


@app.post("/segmenter-form", response_model=SegmentResponse, tags=["Segmentation"])
async def segmenter_form(
        SCRIPT: str = Form(..., description="Le texte à segmenter"),
        NUMBER_IMAGES: str = Form(..., description="Le nombre de segments souhaités")
):
    """
    Segmente un texte - ACCEPTE UNIQUEMENT FORM-DATA

    - **SCRIPT**: Le texte à segmenter (peut contenir des timestamps)
    - **NUMBER_IMAGES**: Le nombre de segments souhaités

    Retourne le texte segmenté avec les segments numérotés (1.segment1 2.segment2 etc.)
    """
    try:
        nb_segments = int(NUMBER_IMAGES)

        if nb_segments <= 0:
            raise HTTPException(
                status_code=400,
                detail="NUMBER_IMAGES doit être un nombre positif"
            )

        resultat_segments, resultat_timestamps = segmenter_texte(SCRIPT, nb_segments)

        return SegmentResponse(
            segments=resultat_segments,
            timestamps=resultat_timestamps or "",
            nombre_segments=nb_segments
        )

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="NUMBER_IMAGES doit être une chaîne représentant un nombre entier valide"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la segmentation: {str(e)}"
        )


@app.post("/segmenter", response_model=SegmentResponse, tags=["Segmentation"])
async def segmenter(
        request: Request,
        SCRIPT: Optional[str] = Form(None),
        NUMBER_IMAGES: Optional[str] = Form(None)
):
    """
    Segmente un texte - ACCEPTE JSON OU FORM-DATA

    - **SCRIPT**: Le texte à segmenter (peut contenir des timestamps)
    - **NUMBER_IMAGES**: Le nombre de segments souhaités

    Retourne le texte segmenté avec les segments numérotés (1.segment1 2.segment2 etc.)
    """
    try:
        # Détecter le type de contenu
        content_type = request.headers.get("content-type", "")

        if "application/json" in content_type:
            # Traiter comme JSON
            body = await request.json()
            script = body.get("SCRIPT")
            number_images = body.get("NUMBER_IMAGES")
        else:
            # Traiter comme form-data
            script = SCRIPT
            number_images = NUMBER_IMAGES

        if not script or not number_images:
            raise HTTPException(
                status_code=400,
                detail="SCRIPT et NUMBER_IMAGES sont requis"
            )

        nb_segments = int(number_images)

        if nb_segments <= 0:
            raise HTTPException(
                status_code=400,
                detail="NUMBER_IMAGES doit être un nombre positif"
            )

        resultat_segments, resultat_timestamps = segmenter_texte(script, nb_segments)

        return SegmentResponse(
            segments=resultat_segments,
            timestamps=resultat_timestamps or "",
            nombre_segments=nb_segments
        )

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="NUMBER_IMAGES doit être une chaîne représentant un nombre entier valide"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la segmentation: {str(e)}"
        )


@app.get("/health", tags=["Health"])
async def health_check():
    """Vérifier l'état de l'API"""
    return {"status": "healthy"}


# Pour lancer l'application en développement
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)