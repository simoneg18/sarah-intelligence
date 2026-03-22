#!/usr/bin/env python3
"""Generate SARAh PDF guide and send via WhatsApp."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT

OUTPUT_PATH = "/Users/simonegozzi/claudecode_outreach/output/guida-comandi-sarah.pdf"

def build_guide():
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        topMargin=2*cm,
        bottomMargin=2*cm,
        leftMargin=2.5*cm,
        rightMargin=2.5*cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Title'],
        fontSize=24, textColor=HexColor('#1a1a2e'),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        'Subtitle', parent=styles['Normal'],
        fontSize=13, textColor=HexColor('#6c63ff'),
        alignment=TA_CENTER, spaceAfter=20,
    )
    heading_style = ParagraphStyle(
        'CustomHeading', parent=styles['Heading2'],
        fontSize=14, textColor=HexColor('#1a1a2e'),
        spaceBefore=16, spaceAfter=8,
    )
    body_style = ParagraphStyle(
        'CustomBody', parent=styles['Normal'],
        fontSize=10.5, leading=15, spaceAfter=6,
    )
    command_style = ParagraphStyle(
        'Command', parent=styles['Normal'],
        fontSize=10, leading=14, textColor=HexColor('#333333'),
        leftIndent=12, spaceAfter=4,
    )
    emoji_style = ParagraphStyle(
        'Emoji', parent=styles['Normal'],
        fontSize=10, leading=14, leftIndent=12, spaceAfter=2,
    )
    footer_style = ParagraphStyle(
        'Footer', parent=styles['Normal'],
        fontSize=9, textColor=HexColor('#888888'),
        alignment=TA_CENTER, spaceBefore=20,
    )

    story = []

    # Title
    story.append(Paragraph("SARAh", title_style))
    story.append(Paragraph("l'unclock intelligence", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor('#6c63ff')))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "SARAh e' la tua assistente AI per l'intelligence su YouTube. "
        "Analizza video, genera briefing audio e te li manda su WhatsApp. "
        "E' meteoropatica: il suo umore cambia in base al meteo di Milano!",
        body_style
    ))
    story.append(Spacer(1, 8))

    # Commands section
    commands = [
        ("1. Analisi canale", "Analizza gli ultimi N video di un creator YouTube.",
         [
             '"ultimi 5 video di Chase"',
             '"riassumimi gli ultimi 3 video di Liam Ottley"',
             '"cosa ha detto Cole Medin su n8n"',
         ]),
        ("2. Video singolo", "Manda un link YouTube e SARAh lo analizza.",
         [
             '"analizza questo https://youtube.com/watch?v=abc123"',
             "Oppure manda direttamente il link senza testo aggiuntivo.",
         ]),
        ("3. Cerca topic", "Scopri chi parla di un argomento su YouTube.",
         [
             '"chi parla di MCP servers?"',
             '"creator italiani che parlano di AI agents questa settimana"',
         ]),
        ("4. Confronto creator", "Confronta cosa dicono piu' creator su un topic.",
         [
             '"confronta Chase e Cole Medin su Claude Code"',
             '"cosa dicono Chase, Liam e Cole su AI agents"',
         ]),
        ("5. Approfondimento", "Approfondisci un punto da un'analisi precedente.",
         [
             '"approfondisci il punto sugli MCP servers"',
             '"dimmi di piu\' sul secondo video"',
         ]),
        ("6. Programmazione", "Imposta briefing ricorrenti automatici.",
         [
             '"aggiornami ogni lunedi\' sui video di Chase"',
             '"briefing settimanale su Cole Medin ogni venerdi\'"',
         ]),
        ("7. Novita'", "Cerca le ultime novita' su un topic senza specificare un creator.",
         [
             '"novita\' su Claude Code"',
             '"cosa c\'e\' di nuovo su AI agents questa settimana"',
         ]),
        ("8. Saluta SARAh", "Saluta SARAh e scopri come sta!",
         [
             '"ciao SARAh"',
             '"buongiorno, come stai?"',
         ]),
    ]

    for title, desc, examples in commands:
        story.append(Paragraph(title, heading_style))
        story.append(Paragraph(desc, body_style))
        for ex in examples:
            story.append(Paragraph(f"<i>{ex}</i>", command_style))
        story.append(Spacer(1, 4))

    # Confirmation flow section
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#cccccc')))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Come funziona la conferma", heading_style))
    story.append(Paragraph(
        "Quando fai una richiesta, SARAh ti mostra subito i video che analizzerebbe "
        "con titolo, link YouTube e tempo stimato. Rispondi <b>si'</b> per procedere "
        "o <b>no</b> per annullare.",
        body_style
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Esempio:", body_style))
    story.append(Paragraph('<i>Tu: "ultimi 2 video di Enkk"</i>', command_style))
    story.append(Paragraph('<i>SARAh: Ecco cosa ho trovato per Enkk:</i>', command_style))
    story.append(Paragraph('<i>  1. Titolo Video 1 - link</i>', command_style))
    story.append(Paragraph('<i>  2. Titolo Video 2 - link</i>', command_style))
    story.append(Paragraph('<i>  Tempo stimato: ~3 minuti</i>', command_style))
    story.append(Paragraph('<i>  Vuoi che proceda? Rispondi si\' o no.</i>', command_style))
    story.append(Paragraph("<i>Tu: \"si'\"</i>", command_style))
    story.append(Paragraph('<i>SARAh: Perfetto! Ci lavoro subito...</i>', command_style))

    # Personality section
    story.append(Spacer(1, 8))
    story.append(Paragraph("La personalita' di SARAh", heading_style))
    story.append(Paragraph(
        "SARAh e' meteoropatica! Il suo umore cambia in base al meteo di Milano:",
        body_style
    ))
    mood_data = [
        ["Meteo", "Umore", ""],
        ["Sole", "Felicissima", "Allegra e carica"],
        ["Nuvoloso / Nebbia", "Cosi' cosi'", "Un po' svogliata ma operativa"],
        ["Pioggia / Temporale", "Un po' triste", "Malinconica ma disponibile"],
        ["Neve", "Al settimo cielo!", "Super entusiasta"],
    ]
    mood_table = Table(mood_data, colWidths=[4.5*cm, 4*cm, 6*cm])
    mood_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#6c63ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#ffffff')),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#f8f8ff'), HexColor('#ffffff')]),
    ]))
    story.append(mood_table)

    # Known creators
    story.append(Spacer(1, 12))
    story.append(Paragraph("Creator conosciuti", heading_style))
    story.append(Paragraph(
        "Chase (Chase H AI), Cole Medin, Liam Ottley, Matt Wolfe, AI Jason, Enkk. "
        "Puoi anche usare qualsiasi nome: SARAh prova a cercarlo automaticamente su YouTube!",
        body_style
    ))

    # Footer
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#cccccc')))
    story.append(Paragraph("SARAh, l'unclock intelligence | unclock.it", footer_style))

    doc.build(story)
    print(f"PDF generato: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    build_guide()
