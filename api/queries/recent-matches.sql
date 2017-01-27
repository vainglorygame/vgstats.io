SELECT ARRAY_TO_JSON(ARRAY(
SELECT
  JSONB_BUILD_OBJECT(
  'id', match.data->>'id',
  'date', match.data->'attributes'->>'createdAt',
  'duration', CAST(
    match.data->'attributes'->>'duration'
    AS INTEGER),
  'teams', ARRAY(
    SELECT(
      SELECT
      JSONB_BUILD_OBJECT(
      'id', roster.data->>'id',
      'side', roster.data->'attributes'->'stats'->>'side',
      'kills', roster.data->'attributes'->'stats'->>'heroKills',
      'players', ARRAY(
        SELECT(
          SELECT
            participant.data->'attributes'->>'actor'
          FROM participant WHERE relparticipant->>'id' = participant.data->>'id')
        FROM JSONB_ARRAY_ELEMENTS(roster.data->'relationships'->'participants'->'data') relparticipant
      )
      )
      FROM roster WHERE relroster->>'id' = roster.data->>'id')
    FROM JSONB_ARRAY_ELEMENTS(match.data->'relationships'->'rosters'->'data') relroster
  )
)
FROM match
ORDER BY match.data->'attributes'->>'createdAt' DESC
)) AS data
