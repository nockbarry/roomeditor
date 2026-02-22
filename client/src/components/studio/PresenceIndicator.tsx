import { useCollabStore, type RemoteUser } from "../../stores/collabStore.ts";
import { Users } from "lucide-react";

/**
 * Small colored dots in the top bar showing connected users.
 */
export function PresenceIndicator() {
  const connected = useCollabStore((s) => s.connected);
  const users = useCollabStore((s) => s.users);

  if (!connected || users.length === 0) return null;

  return (
    <div className="flex items-center gap-1.5 bg-gray-900/80 backdrop-blur-sm rounded-full px-2.5 py-1 border border-gray-700/50">
      <Users className="w-3 h-3 text-gray-500" />
      <div className="flex -space-x-1">
        {users.map((user) => (
          <div
            key={user.user_id}
            title={`User ${user.user_id}`}
            className="w-4 h-4 rounded-full border-2 border-gray-900"
            style={{ backgroundColor: user.color }}
          />
        ))}
      </div>
      <span className="text-[10px] text-gray-500">{users.length}</span>
    </div>
  );
}
