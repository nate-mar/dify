export enum TracingProvider {
  langSmith = 'langsmith',
  langfuse = 'langfuse',
  opik = 'opik',
  arizePhoenix = 'arizePhoenix',
}

export type LangSmithConfig = {
  api_key: string
  project: string
  endpoint: string
}

export type LangFuseConfig = {
  public_key: string
  secret_key: string
  host: string
}

export type OpikConfig = {
  api_key: string
  project: string
  workspace: string
  url: string
}

export type ArizePhoenixConfig = {
  api_key: string;
  project: string;
  host: string;
  protocol: string;
  space_key: string;
}
