SELECT
  predicted_roles.participant_id,
  (SELECT
      JSONB_BUILD_ARRAY(key, (value->>0)::float)
    FROM
      JSONB_EACH(data)
    ORDER BY (value->>0)::float DESC
    LIMIT 1
  ),
  (SELECT
    JSONB_BUILD_OBJECT(
      'actor', participant.data->'attributes'->'actor',
      'items', participant.data->'attributes'->'stats'->'items'
    )
   FROM participant WHERE participant.id=predicted_roles.participant_id) AS actor
FROM predicted_roles
LIMIT 100