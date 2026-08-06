SELECT table_name, string_agg(privilege_type, ', ')
FROM information_schema.role_table_grants
WHERE grantee = 'authenticated' AND table_schema = 'public'
GROUP BY table_name ORDER BY 1;